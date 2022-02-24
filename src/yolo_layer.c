#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

extern int check_mistakes;

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)xcalloc(1, sizeof(float));
    l.biases = (float*)xcalloc(total * 2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = (int*)xcalloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truth_size = 4 + 2;
    l.truths = l.max_boxes*l.truth_size;    // 90*(4 + 1);
    l.labels = (int*)xcalloc(batch * l.w*l.h*l.n, sizeof(int));
    for (i = 0; i < batch * l.w*l.h*l.n; ++i) l.labels[i] = -1;
    l.class_ids = (int*)xcalloc(batch * l.w*l.h*l.n, sizeof(int));
    for (i = 0; i < batch * l.w*l.h*l.n; ++i) l.class_ids[i] = -1;

    l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.output_avg_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }
#endif

    fprintf(stderr, "yolo\n");
    srand(time(0));

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    if (l->embedding_output) l->embedding_output = (float*)xrealloc(l->output, l->batch * l->embedding_size * l->n * l->h * l->w * sizeof(float));
    if (l->labels) l->labels = (int*)xrealloc(l->labels, l->batch * l->n * l->h * l->w * sizeof(int));
    if (l->class_ids) l->class_ids = (int*)xrealloc(l->class_ids, l->batch * l->n * l->h * l->w * sizeof(int));

    if (!l->output_pinned) l->output = (float*)xrealloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)xrealloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        CHECK_CUDA(cudaFreeHost(l->output));
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        CHECK_CUDA(cudaFreeHost(l->delta));
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->output_avg_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
    l->output_avg_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, int new_coords)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
    // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
    // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    if (new_coords) {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
        b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
    }
    else {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    }
    return b;
}

static inline float fix_nan_inf(float val)
{
    if (isnan(val) || isinf(val)) val = 0;
    return val;
}

static inline float clip_value(float val, const float max_val)
{
    if (val > max_val) {
        //printf("\n val = %f > max_val = %f \n", val, max_val);
        val = max_val;
    }
    else if (val < -max_val) {
        //printf("\n val = %f < -max_val = %f \n", val, -max_val);
        val = -max_val;
    }
    return val;
}

ious delta_yolo_box(int track_id, int num_elems, int update_delta, box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss, int accumulate, float max_delta, int *rewritten_bbox, int new_coords)
{
    if (update_delta) {
        if (delta[index + 0 * stride] || delta[index + 1 * stride] || delta[index + 2 * stride] || delta[index + 3 * stride]) {
            (*rewritten_bbox)++;
        }
    }

    box ltruth;
    ltruth.x = truth.x;
    ltruth.y = truth.y;
    ltruth.w = truth.w;
    ltruth.h = truth.h;

//    printf("\n\nPOINT HERE2 %f %f\n\n", truth.w, truth.h);

//    printf("W: %f H: %f\n", truth.w, truth.h);

//    if ((truth.h < 0.0002) && (truth.w < 0.0002)){
//        ltruth.h = 0.06;
//        ltruth.w = 0.051;

//        printf("\n\nPOINT HERE %f %f\n\n", truth.w, truth.h);
//    }

    pid_t pid = syscall(__NR_gettid);

    ious all_ious = { 0 };
    // i - step in grid width
    // j - step in grid height
    //  Returns a box in absolute coordinates
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);
    all_ious.iou = box_iou(pred, ltruth);
    all_ious.giou = box_giou(pred, ltruth);
    all_ious.diou = box_diou(pred, ltruth);
    all_ious.ciou = box_ciou(pred, ltruth);

    printf("(%x)PREDICTION: %f %f %f %f\n", pid, pred.x, pred.y, pred.w, pred.h);

//    int s_id = (track_id/1000000)%10;
    int s_id, img_id, label_id;
    char* fname_base = (char*)malloc(100*sizeof(char));

//    if( access( "current_batch.txt", F_OK ) == 0 ) {
//        FILE* fbatch = fopen("current_batch.txt", "r");

//        while(fscanf(fbatch, "%d %d %d %s\n", &img_id, &s_id, &label_id, fname_base)==4){
////            fscanf(fbatch, "%d %d %d %s\n", &img_id, &s_id, &label_id, fname_base);
//            if(track_id == img_id) {
//                break;
//            }
//        }

//        fclose(fbatch);
//    }

    char* fname_hash;
    fname_hash = (char*)malloc(100*sizeof (char));

    sprintf(fname_hash, "hash/%d.txt", track_id);
    if( access( fname_hash, F_OK ) == 0 ) {
        FILE* fhash = fopen(fname_hash, "r");
        if(fscanf(fhash, "%d %d %s\n", &s_id, &label_id, fname_base) != 3) {
            printf("HASH FILE DELTA INCORRECT %d %d %d %s\n", track_id, s_id, label_id, fname_base);
            all_ious.iou = all_ious.giou = all_ious.ciou = all_ious.diou = 0;
//            exit(0);
            return all_ious;
        }
        fclose(fhash);
    } else {
        printf("NO FILE HASH DELTA FOUND %d\n", track_id);
//        all_ious = {0};
        all_ious.iou = all_ious.giou = all_ious.ciou = all_ious.diou = 0;
//        exit(0);
        return all_ious;
    }

    free(fname_hash);
//    free(fname_base);

    float alpha = 0.75f, beta = 0.5f;

    if (update_delta) {

        typedef struct pred_args {
            int id;
            int cnt;
            int lid; //label_id
            float x;
            float y;
            float w;
            float h;
            float loss;
            float conf;
        } pred_args_t;

//        if (s_id == 1) {
//        if ((s_id == 1) || (s_id == 2)) {
        if (s_id > 0) {
//            int img_id = (track_id)%1000000;
//            int label_id = (track_id)/10000000;

            float den = sqrt(pred.w*pred.w/4 + pred.h*pred.h/4);
            float num = -sqrt((pred.x-truth.x)*(pred.x-truth.x) + (pred.y-truth.y)*(pred.y-truth.y));
            float center_iou = den > 2*FLT_EPSILON ? (1-exp(num/den)) : 0;

            // 1 if centers coincide

            if(s_id == 1 ) {
                all_ious.iou = center_iou;
            } else if ((s_id == 2) || (s_id == 4)){
                all_ious.iou *= alpha;
                all_ious.iou += center_iou*(1-alpha);
//                all_ious.iou /= 2.0f;
            } else if (s_id == 3) {
                all_ious.iou = 0.0;
            }


            char* fname_init;
            fname_init = (char*)malloc(100*sizeof (char));
//            sprintf(fname_init, "/home/fazevedo/Desktop/PhD/COCOPerson/labels/train/%012d_init.txt", img_id);
            sprintf(fname_init, "%s_init.txt", fname_base);
            printf("\n(%x)NAME OF INIT FILE %s %d\n", pid, fname_init, label_id);

            char* fname_scale;
            fname_scale = (char*)malloc(100*sizeof (char));
//            sprintf(fname_scale, "/home/fazevedo/Desktop/PhD/COCOPerson/labels/train/%012d_scale.txt", img_id);
            sprintf(fname_scale, "%s_scale.txt", fname_base);
            printf("(%x)NAME OF SCALE FILE %s %d\n", pid, fname_scale, label_id);

            char* fname_pred;
            fname_pred = (char*)malloc(100*sizeof (char));
//            sprintf(fname_pred, "/home/fazevedo/Desktop/PhD/COCOPerson/labels/train/%012d_pred.txt", img_id);
            sprintf(fname_pred, "%s_pred.txt", fname_base);
            printf("(%x)NAME OF PREDICTION FILE %s %d\n", pid, fname_pred, label_id);

            if( access( fname_init, F_OK ) == 0 ) {
                FILE* finit = fopen(fname_init, "r");

                int z = 0;

                float bh, bw, x_init, y_init;
                int id = 0;

                while(fscanf(finit, "%d %f %f %f %f", &id, &x_init, &y_init, &bw, &bh) == 5) {
                    if (z != label_id){ // Verify if it is sample ID needed.
                        z++;
                        continue;
                    }
                    break;
                }
                fclose(finit);

                if( access( fname_scale, F_OK ) == 0 ) {
                    FILE* fscale = fopen(fname_scale, "r");

                    float dx, dy, sw, sh;
                    int read, flip;

                    if(fscanf(fscale, "%d %f %f %f %f %d", & read, &dx, &dy, &sw, &sh, &flip) == 6) {
                        printf("(%x)SCALES: %f %f %f %f %f %f %d\n", pid, x_init, y_init, dx, dy, sw, sh, flip);

                        float x_rec = (fabs(flip - truth.x) + dx)/sw;
                        bw = pred.w/sw;
                        bw += 2*fabs(x_rec-x_init);
                        bw = fminf(1.0f, bw);

                        float y_rec = (truth.y + dy)/sh;
                        bh = pred.h/sh;
                        bh += 2*fabs(y_rec-y_init);
                        bh = fminf(1.0f, bh);

                        float xpred_rec = (fabs(flip - pred.x) + dx)/sw;
                        float ypred_rec = (pred.y + dy)/sh;

                        printf("(%x)RECOVER: %f %f %f %f\n", pid, x_rec, y_rec, w, h);
                        printf("(%x)RECOVERPRED: %f %f %f %f X:%f Y:%f TX:%f TY:%f XI:%f YI:%f\n", pid, xpred_rec, ypred_rec, w, h, pred.x, pred.y, truth.x, truth.y, x_init, y_init);

//                        float den = sqrt(pred.w*pred.w/4 + pred.h*pred.h/4);
//                        float num = -sqrt((pred.x-truth.x)*(pred.x-truth.x) + (pred.y-truth.y)*(pred.y-truth.y));
//                        float center_iou = den > 2*FLT_EPSILON ? (1-exp(num/den)) : 1;
                        printf("(%x)CENTER LOSS: %f\n", pid, center_iou);

                        int conf_index = index + 4*lw*lh;

//                        FILE* fpred = fopen(fname_pred, "a");
//                        fprintf(fpred, "%d %.7f %.7f %.7f %.7f %.7f %.7f\n", 0, x_rec, y_rec, w, h, center_iou, x[conf_index]);
//                        fclose(fpred);

                        float ax, ay, ah, aw, aiou, aconf;
                        int aid, cnt, alid;
                        int found = 0;

                        pred_args_t *buffer;
//                        char* buffer;
                        buffer = (pred_args_t*)malloc(1*sizeof (pred_args_t));
                        int n_char = 0, line_size=67;

                        float best_iou = 0.0f;
                        int line_nr = -1, line_cnt = 0;
                        box pred_shift = pred;
                        if(s_id < 3) {
//                            pred_shift.x = pred_shift.y = 0.0f;
                            pred_shift.x = x_init;
                            pred_shift.y = y_init;
                        } else {
                            pred_shift.x = xpred_rec;
                            pred_shift.y = ypred_rec;
                        }

                        pred_shift.h = bh;
                        pred_shift.w = bw;

                        box read_box = {0};

                        if( access( fname_pred, F_OK ) == 0 ) {
//                            printf("\n\n\n FOUNNNNNND \n\n\n\n");

                            FILE* fpred = fopen(fname_pred, "r");

                            while(fscanf(fpred, "%d %d %f %f %f %f %f %f %d\n", &aid, &alid, &ax, &ay, &aw, &ah, &aiou, &aconf, &cnt) == 9) {
                                if ((aid == id) &&
                                    (label_id == alid)){
//                                    (fabs(x_init-ax) < 2*FLT_EPSILON) &&
//                                    (fabs(y_init-ay) < 2*FLT_EPSILON) ){
//                                    found = 1;
                                    read_box.x = ax;
                                    read_box.y = ay;
                                    read_box.w = aw;
                                    read_box.h = ah;

                                    float iou = box_iou(pred_shift, read_box);

                                    if ((iou > 0.75f) && (iou > best_iou)) {
//                                    if ((iou > best_iou)) {
                                        line_nr = line_cnt;
                                        best_iou = iou;
                                        found = 1;
                                    }
                                }

                                buffer = (pred_args_t*)xrealloc(buffer, (line_cnt+1)*sizeof(pred_args_t));

                                buffer[line_cnt].id   = aid;
                                buffer[line_cnt].lid  = alid;
                                buffer[line_cnt].x    = ax;
                                buffer[line_cnt].y    = ay;
                                buffer[line_cnt].w    = aw;
                                buffer[line_cnt].h    = ah;
                                buffer[line_cnt].loss = aiou;
                                buffer[line_cnt].conf = aconf;
                                buffer[line_cnt].cnt  = cnt;

                                line_cnt++;
                            }
                            fclose(fpred);

                            if (line_nr > -1) { // Found similar
                                buffer[line_nr].cnt++;
                                if(s_id==3) { // Only label. Working with predictions
                                    buffer[line_nr].x += (xpred_rec-buffer[line_nr].x)/buffer[line_nr].cnt;
                                    buffer[line_nr].y += (ypred_rec-buffer[line_nr].y)/buffer[line_nr].cnt;
                                }
                                buffer[line_nr].w += (bw-buffer[line_nr].w)/buffer[line_nr].cnt;
                                buffer[line_nr].h += (bh-buffer[line_nr].h)/buffer[line_nr].cnt;
                                buffer[line_nr].loss += (center_iou-buffer[line_nr].loss)/buffer[line_nr].cnt;
                                buffer[line_nr].conf += (x[conf_index]-buffer[line_nr].conf)/buffer[line_nr].cnt;
                            } else {
                                buffer = (pred_args_t*)xrealloc(buffer, (line_cnt+1)*sizeof(pred_args_t));

                                buffer[line_cnt].id   = id;
                                buffer[line_cnt].lid  = label_id;
                                buffer[line_cnt].x    = x_init;
                                buffer[line_cnt].y    = y_init;
                                if(s_id==3) { // Only label. Working with predictions
                                    buffer[line_cnt].x    = xpred_rec;
                                    buffer[line_cnt].y    = ypred_rec;
                                }
                                buffer[line_cnt].w    = bw;
                                buffer[line_cnt].h    = bh;
                                buffer[line_cnt].loss = center_iou;
                                buffer[line_cnt].conf = x[conf_index];
                                buffer[line_cnt].cnt  = 1;

                                line_cnt++;
                            }

                            for (int k = 0; k < line_cnt-1; k++) {
                                if(buffer[k].cnt == 0) {
                                    continue;
                                }

                                box box1 = {0};
                                box1.x = buffer[k].x;
                                box1.y = buffer[k].y;
                                box1.w = buffer[k].w;
                                box1.h = buffer[k].h;

                                for (int l = k+1; l <line_cnt; l++) {

                                    if((buffer[k].id == buffer[l].id) && (buffer[k].lid == buffer[l].lid)) {

                                        box box2 = {0};
                                        box2.x = buffer[l].x;
                                        box2.y = buffer[l].y;
                                        box2.w = buffer[l].w;
                                        box2.h = buffer[l].h;

                                        float iou = box_iou(box1, box2);
                                        if(iou > 0.75f) { //Merge boxes
                                            buffer[k].x = (buffer[k].x*buffer[k].cnt + buffer[l].x*buffer[l].cnt)/(buffer[k].cnt + buffer[l].cnt);
                                            buffer[k].y = (buffer[k].y*buffer[k].cnt + buffer[l].y*buffer[l].cnt)/(buffer[k].cnt + buffer[l].cnt);
                                            buffer[k].w = (buffer[k].w*buffer[k].cnt + buffer[l].w*buffer[l].cnt)/(buffer[k].cnt + buffer[l].cnt);
                                            buffer[k].h = (buffer[k].h*buffer[k].cnt + buffer[l].h*buffer[l].cnt)/(buffer[k].cnt + buffer[l].cnt);
                                            buffer[k].loss = (buffer[k].loss*buffer[k].cnt + buffer[l].loss*buffer[l].cnt)/(buffer[k].cnt + buffer[l].cnt);
                                            buffer[k].conf = (buffer[k].conf*buffer[k].cnt + buffer[l].conf*buffer[l].cnt)/(buffer[k].cnt + buffer[l].cnt);
                                            buffer[k].cnt += buffer[l].cnt;
                                            buffer[l].cnt = 0;
                                            printf("\n\n(%x)BOX MERGED %s\n\n", pid, fname_base);
                                        }

                                    }
                                }
                            }






//                            fpred = fopen(fname_pred, "r");

//                            line_cnt = 0;
//                            while(fscanf(fpred, "%d %f %f %f %f %f %f %d\n", &aid, &ax, &ay, &aw, &ah, &aloss, &aconf, &cnt) == 8) {
//                                n_char+=line_size; // All line plus \n
//                                buffer = realloc(buffer, n_char+1);

//                                if (line_nr == line_cnt){
//                                    cnt++;
//                                    sprintf(buffer+(n_char-line_size), "%d %.7f %.7f %.7f %.7f %.7f %.7f %d\n", id, x_init, y_init, aw + (w-aw)/cnt, ah + (h-ah)/cnt, aloss + (center_iou-aloss)/cnt, aconf + (x[conf_index]-aconf)/cnt, cnt);
//                                }
//                                else {
//                                    sprintf(buffer+(n_char-line_size), "%d %.7f %.7f %.7f %.7f %.7f %.7f %d\n", aid, ax, ay, aw, ah, aloss, aconf, cnt);
//                                }
//                                line_cnt++;
//                            }
//                            fclose(fpred);

//                            if (found == 0){
//                                // ADD Line
//                                cnt = 1;
//                                n_char+=line_size; // All line plus \n
//                                buffer = realloc(buffer, n_char+1);
//                                sprintf(buffer+(n_char-line_size), "%d %.7f %.7f %.7f %.7f %.7f %.7f %d\n", id, x_init, y_init, w, h, center_iou, x[conf_index], cnt);
////                                        fprintf(fw, "%d %.7f %.7f %.7f %.7f\n", sid, sx, sy, sw, sh);
//                            }

//                            printf("%d\n %s \n\n\n\n", line_cnt, buffer);

                            FILE* fw = fopen(fname_pred, "w");
//                            buffer[n_char] = '\0';
//                            fprintf(fw, "%s", buffer);
                            for (int iter = 0; iter<line_cnt; iter++)
                            {
                                if(buffer[iter].cnt > 0) {
                                    fprintf(fw, "%d %d %.7f %.7f %.7f %.7f %.7f %.7f %d\n", buffer[iter].id, buffer[iter].lid, buffer[iter].x, buffer[iter].y, buffer[iter].w, buffer[iter].h, buffer[iter].loss, buffer[iter].conf, buffer[iter].cnt);
                                }
                            }
                            fclose(fw);
                            free(buffer);
                        } else {
                            // file doesn't exist
                            printf("FILE PRED DOES NOT EXIST\n");
                            cnt = 1;
                            FILE* fw = fopen(fname_pred, "w");
                            fprintf(fw, "%d %d %.7f %.7f %.7f %.7f %.7f %.7f %d\n", id, label_id, (s_id==3 ? xpred_rec : x_init), (s_id==3 ? ypred_rec : y_init), bw, bh, center_iou, x[conf_index], cnt);
                            fclose(fw);
                        }
                    }

                    fclose(fscale);

//                    fscale = fopen(fname_scale, "w");
//                    fprintf(fscale, "%d %.7f %.7f %.7f %.7f %d\n", 1, dx, dy, sw, sh, flip);
//                    fclose(fscale);
                }
            }

            free(fname_init);
            free(fname_scale);
            free(fname_pred);

        }

    }

    free(fname_base);



//    b.y = (j + x[index + 1 * stride])
//    printf("BOX PRED: %f %f %f %f %f %d\n", pred.x, pred.y, pred.w, pred.h, x[index + 4 * stride], stride);
//    pred = get_yolo_box(x, biases, n, index+1, i, j, lw, lh, w, h, stride, new_coords);
//    printf("BOX PRED3: %f %f %f %f %f %d\n", pred.x, pred.y, pred.w, pred.h, x[index+1 + 4 * stride], stride);

//    printf("\n\n\n\n\nVALUES %f %f %f %f %f %f %f %f %f %d\n", truth.x, truth.y, pred.x, pred.y, truth.w, truth.h, pred.w, pred.h, all_ious.iou, iou_loss);

//    printf("VAL X: %f VAL Y: %f\n", fabs(pred.x-truth.x)/truth.w, fabs(pred.y-truth.y)/truth.h);

    float ex = fabs(pred.x-truth.x)/pred.w, ey = fabs(pred.y-truth.y)/pred.h;
    all_ious.ciou = sqrt(ex*ex+ey*ey);

//    if(all_ious.iou > 2*FLT_EPSILON) {
    if(0) {
        all_ious.ciou = 1000;

//        float covpredw  = (pred.w/4.0f)*(pred.w/4.0f);
//        float covtruthw = (truth.w/4.0f)*(truth.w/4.0f);
//        float covpredh  = (pred.h/4.0f)*(pred.h/4.0f);
//        float covtruthh = (truth.h/4.0f)*(truth.h/4.0f);

        // avoid nan in dx_box_iou
        if (pred.w < 2*FLT_EPSILON) { pred.w = 1.0; }
        if (pred.h < 2*FLT_EPSILON) { pred.h = 1.0; }

        double covpredw  = (pred.w*pred.w)/16.0;
        double covtruthw = (truth.w*truth.w)/16.0;
        double covpredh  = (pred.h*pred.h)/16.0;
        double covtruthh = (truth.h*truth.h)/16.0;

//        printf("COVPW: %f\n", pred.w/4.0);
//        printf("COVPH: %f\n", pred.h/4.0);
//        printf("COVTW: %f\n", truth.w/4.0);
//        printf("COVTH: %f\n", truth.h/4.0);

//        printf("PX: %f\n", pred.x);
//        printf("PY: %f\n", pred.y);
//        printf("TX: %f\n", truth.x);
//        printf("TY: %f\n", truth.y);

        box medbox;
        medbox.x = (pred.x + truth.x)/2.0;
        medbox.y = (pred.y + truth.y)/2.0;

        double covmedw = (covpredw + covtruthw)/2.0;
        double covmedh = (covpredh + covtruthh)/2.0;


        ////// KL LOSSSSS
//        double kldiv1 = log(covtruthw/(covpredw+FLT_EPSILON));//log(covtruthw/covpredw);
        float kldiv1 = log((truth.w*truth.w)/((pred.w*pred.w)+FLT_EPSILON));
        kldiv1 -= 1;
        kldiv1 += (pred.x-truth.x)*(pred.x-truth.x)/(covtruthw+FLT_EPSILON);
        kldiv1 += (pred.w*pred.w)/((truth.w*truth.w)+FLT_EPSILON);
        kldiv1 /= 2;

        if ((kldiv1 > 500) || isnan(kldiv1) || isinf(kldiv1)){
            kldiv1 = 500;
            printf("BAD VALUES KL1: %f\n", kldiv1);
            printf("TRUTH: %f %f %f %f\nPRED: %f %f %f %f", truth.x, truth.y, truth.w, truth.h, pred.x, pred.y, pred.w, pred.h);

        }

        float kldiv2 = log((truth.h*truth.h)/((pred.h*pred.h)+FLT_EPSILON));//log(covtruthh/covpredh);
        kldiv2 -= 1;
        kldiv2 += (pred.y-truth.y)*(pred.y-truth.y)/(covtruthh+FLT_EPSILON);
        kldiv2 += (pred.h*pred.h)/((truth.h*truth.h)+FLT_EPSILON);
        kldiv2 /= 2;

        if ((kldiv2 > 500) || isnan(kldiv2) || isinf(kldiv2)){
            kldiv2 = 500;
            printf("BAD VALUES KL2: %f\n", kldiv2);
            printf("TRUTH: %f %f %f %f\nPRED: %f %f %f %f", truth.x, truth.y, truth.w, truth.h, pred.x, pred.y, pred.w, pred.h);

        }
        double kldiv = kldiv1 + kldiv2;
        ////// KL LOSSSSS

//        kldiv1 = log(covpredw+FLT_EPSILON) - log(covtruthw+FLT_EPSILON);//log(covtruthw/covpredw);
//        kldiv1 -= 1;        kldiv /= 2;

//        kldiv1 += (truth.x-pred.x)*(truth.x-pred.x)/(covpredw+FLT_EPSILON);
//        kldiv1 += covtruthw/(covpredw+FLT_EPSILON);
//        kldiv1 /= 2;

//        kldiv2 = log(covpredh+FLT_EPSILON) - log(covtruthh+FLT_EPSILON);//log(covtruthh/covpredh);
//        kldiv2 -= 1;
//        kldiv2 += (truth.y-pred.y)*(truth.y-pred.y)/(covpredh+FLT_EPSILON);
//        kldiv2 += covtruthh/(covpredh+FLT_EPSILON);
//        kldiv2 /= 2;

//        kldiv += kldiv1 + kldiv2;
//        kldiv /= 2;


        ////// JS LOSSSSS
//        double kldiv1 = log(covmedw/(covpredw+FLT_EPSILON));//log(covtruthw/covpredw);
//        kldiv1 -= 1;
//        kldiv1 += (pred.x-medbox.x)*(pred.x-medbox.x)/(covmedw+FLT_EPSILON);
//        kldiv1 += covpredw/(covmedw+FLT_EPSILON);
//        kldiv1 /= 2;

//        if ((kldiv1 > 500) || isnan(kldiv1) || isinf(kldiv1)){
//            kldiv1 = 500;
//            printf("BAD VALUES KL1: %f\n", kldiv1);
//            printf("TRUTH: %f %f %f %f\nPRED: %f %f %f %f", truth.x, truth.y, truth.w, truth.h, pred.x, pred.y, pred.w, pred.h);

//        }

//        double kldiv2 = log(covmedh/(covpredh+FLT_EPSILON));//log(covtruthh/covpredh);
//        kldiv2 -= 1;
//        kldiv2 += (pred.y-medbox.y)*(pred.y-medbox.y)/(covmedh+FLT_EPSILON);
//        kldiv2 += covpredh/(covmedh+FLT_EPSILON);
//        kldiv2 /= 2;

//        if ((kldiv2 > 500) || isnan(kldiv2) || isinf(kldiv2)){
//            kldiv2 = 500;
//            printf("BAD VALUES KL2: %f\n", kldiv2);
//            printf("TRUTH: %f %f %f %f\nPRED: %f %f %f %f", truth.x, truth.y, truth.w, truth.h, pred.x, pred.y, pred.w, pred.h);

//        }

//        kldiv1 = fmax(0.0, kldiv1);
//        kldiv2 = fmax(0.0, kldiv2);

//        double kldiv = kldiv1 + kldiv2;
////        double kldiv = 1 - exp(-3*(kldiv1 + kldiv2));
////        double kldiv = 1 - exp(-(kldiv1 + kldiv2));

//        kldiv1 = log(covmedw/(covtruthw+FLT_EPSILON));//log(covtruthw/covpredw);
//        kldiv1 -= 1;
//        kldiv1 += (truth.x-medbox.x)*(truth.x-medbox.x)/(covmedw+FLT_EPSILON);
//        kldiv1 += covtruthw/(covmedw+FLT_EPSILON);
//        kldiv1 /= 2;

//        if ((kldiv1 > 500) || isnan(kldiv1) || isinf(kldiv1)){
//            kldiv1 = 500;
//            printf("BAD VALUES KL1: %f\n", kldiv1);
//            printf("TRUTH: %f %f %f %f\nPRED: %f %f %f %f", truth.x, truth.y, truth.w, truth.h, pred.x, pred.y, pred.w, pred.h);

//        }

//        kldiv2 = log(covmedh/(covtruthh+FLT_EPSILON));//log(covtruthh/covpredh);
//        kldiv2 -= 1;
//        kldiv2 += (truth.y-medbox.y)*(truth.y-medbox.y)/(covmedh+FLT_EPSILON);
//        kldiv2 += covtruthh/(covmedh+FLT_EPSILON);
//        kldiv2 /= 2;

//        if ((kldiv2 > 500) || isnan(kldiv2) || isinf(kldiv2)){
//            kldiv2 = 500;
//            printf("BAD VALUES KL2: %f\n", kldiv2);
//            printf("TRUTH: %f %f %f %f\nPRED: %f %f %f %f", truth.x, truth.y, truth.w, truth.h, pred.x, pred.y, pred.w, pred.h);

//        }

//        kldiv1 = fmax(0.0, kldiv1);
//        kldiv2 = fmax(0.0, kldiv2);

//        kldiv += kldiv1 + kldiv2;
////        kldiv += 1 - exp(-3*(kldiv1 + kldiv2));

////        kldiv = sqrt(kldiv/2.0);
//        kldiv /= log(2);
//        kldiv = sqrtf(kldiv/2.0);
    ////// JS LOSSSSS


        if (kldiv > FLT_MAX)
            kldiv = FLT_MAX;

//        printf("VALUES KL DIV XW: %f YH: %f TOT: %f IOUL: %f UP: %d\n", kldiv1, kldiv2, kldiv, 1.0-all_ious.iou, update_delta);


        all_ious.ciou = 10*(float)kldiv;
//        if (all_ious.iou < 0.0f)
//            all_ious.iou = 0.0f;


//        float dx = 16*(truth.x - pred.x)/(truth.w*truth.w);
//        float dy = 16*(truth.y - pred.y)/(truth.h*truth.h);
//        float dw = 1/pred.w - pred.w/(truth.w*truth.w);
//        float dh = 1/pred.h - pred.h/(truth.h*truth.h);

//        dx *= iou_normalizer;
//        dy *= iou_normalizer;
//        dw *= iou_normalizer;
//        dh *= iou_normalizer;

//        if (!accumulate) {
//            delta[index + 0 * stride] = 0;
//            delta[index + 1 * stride] = 0;
//            delta[index + 2 * stride] = 0;
//            delta[index + 3 * stride] = 0;
//        }

//        // accumulate delta
//        delta[index + 0 * stride] += dx;
//        delta[index + 1 * stride] += dy;
//        delta[index + 2 * stride] += dw;
//        delta[index + 3 * stride] += dh;
    }

    if (update_delta == 0)
        return all_ious;

    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }

    float scale_n_el = 1.f;
    if (num_elems > 1) {
        scale_n_el /= num_elems*10;
    }

    if (iou_loss == MSE)    // old loss
//    if(0)
    {
        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2 * n]);
        float th = log(truth.h*h / biases[2 * n + 1]);

//        if ((s_id > 2)) {
//            tx = (pred.x*lw - i);
//            ty = (pred.y*lh - j);
//        }

        if (new_coords) {
            //tx = (truth.x*lw - i + 0.5) / 2;
            //ty = (truth.y*lh - j + 0.5) / 2;
            tw = sqrt(truth.w*w / (4 * biases[2 * n]));
            th = sqrt(truth.h*h / (4 * biases[2 * n + 1]));
        }

        //printf(" tx = %f, ty = %f, tw = %f, th = %f \n", tx, ty, tw, th);
        //printf(" x = %f, y = %f, w = %f, h = %f \n", x[index + 0 * stride], x[index + 1 * stride], x[index + 2 * stride], x[index + 3 * stride]);

        // accumulate delta

        if ((s_id == 4)) {
            delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer * (alpha);// * scale_n_el;
            delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer * (alpha);// * scale_n_el;
        } else if (s_id < 3) {
            delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;// * scale_n_el;
            delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;// * scale_n_el;
        }

        if (s_id == 0){
            delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;// * scale_n_el * (num_elems==1);
            delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;// * scale_n_el * (num_elems==1);
        }
        else if ((s_id == 2) || (s_id == 4)) {
            delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer*alpha;// * scale_n_el * (num_elems==1);
            delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer*alpha;
//            float dtop = ((pred.y-pred.h/2) - truth.y);
//            float dbottom = (truth.y - (pred.y+pred.h/2));
//            if(dtop > 0.0f) { // Above box
//                delta[index + 3 * stride] += sqrt(dtop*h / (4 * biases[2 * n + 1]));
//            } else if(dbottom > 0.0f) { // Below box
//                delta[index + 3 * stride] += sqrt(dbottom*h / (4 * biases[2 * n + 1]));
//            }

//            float dleft  = ((pred.x-pred.w/2) - truth.x);
//            float dright = (truth.x - (pred.x+pred.w/2));

//            if(dleft > 0.0f) { // Above box
//                delta[index + 2 * stride] += sqrt(dleft*w / (4 * biases[2 * n + 1]));
//            } else if(dright > 0.0f) { // Below box
//                delta[index + 2 * stride] += sqrt(dright*w / (4 * biases[2 * n + 1]));
//            }
        }
    }
    else{
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        //float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        //float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;


        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        if (new_coords) {
            //dw *= 8 * x[index + 2 * stride];
            //dh *= 8 * x[index + 3 * stride];
            //dw *= 8 * x[index + 2 * stride] * biases[2 * n] / w;
            //dh *= 8 * x[index + 3 * stride] * biases[2 * n + 1] / h;

            //float grad_w = 8 * exp(-x[index + 2 * stride]) / pow(exp(-x[index + 2 * stride]) + 1, 3);
            //float grad_h = 8 * exp(-x[index + 3 * stride]) / pow(exp(-x[index + 3 * stride]) + 1, 3);
            //dw *= grad_w;
            //dh *= grad_h;
        }
        else {
            dw *= exp(x[index + 2 * stride]);
            dh *= exp(x[index + 3 * stride]);
        }


        //dw *= exp(x[index + 2 * stride]);
        //dh *= exp(x[index + 3 * stride]);

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;


        dx = fix_nan_inf(dx);
        dy = fix_nan_inf(dy);
        dw = fix_nan_inf(dw);
        dh = fix_nan_inf(dh);

        if (max_delta != FLT_MAX) {
            dx = clip_value(dx, max_delta);
            dy = clip_value(dy, max_delta);
            dw = clip_value(dw, max_delta);
            dh = clip_value(dh, max_delta);
        }


        if (!accumulate) {
            delta[index + 0 * stride] = 0;
            delta[index + 1 * stride] = 0;
            delta[index + 2 * stride] = 0;
            delta[index + 3 * stride] = 0;
        }

        // accumulate delta
        delta[index + 0 * stride] += dx*scale_n_el;
        delta[index + 1 * stride] += dy*scale_n_el;
        delta[index + 2 * stride] += dw*scale_n_el;
        delta[index + 3 * stride] += dh*scale_n_el;
    }

    return all_ious;
}

void averages_yolo_deltas(int class_index, int box_index, int stride, int classes, float *delta)
{

    int classes_in_one_box = 0;
    int c;
    for (c = 0; c < classes; ++c) {
        if (delta[class_index + stride*c] > 0) classes_in_one_box++;
    }

    if (classes_in_one_box > 0) {
        delta[box_index + 0 * stride] /= classes_in_one_box;
        delta[box_index + 1 * stride] /= classes_in_one_box;
        delta[box_index + 2 * stride] /= classes_in_one_box;
        delta[box_index + 3 * stride] /= classes_in_one_box;
    }
}

void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers, float cls_normalizer)
{
    int n;
    if (delta[index + stride*class_id]){
        float y_true = 1;
        if(label_smooth_eps) y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
        float result_delta = y_true - output[index + stride*class_id];
        if(!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*class_id] = result_delta;
        //delta[index + stride*class_id] = 1 - output[index + stride*class_id];

        if (classes_multipliers) delta[index + stride*class_id] *= classes_multipliers[class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            float y_true = ((n == class_id) ? 1 : 0);
            if (label_smooth_eps) y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
            float result_delta = y_true - output[index + stride*n];
            if (!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*n] = result_delta;

            if (classes_multipliers && n == class_id) delta[index + stride*class_id] *= classes_multipliers[class_id] * cls_normalizer;
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

int compare_yolo_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh)
{
    int j;
    for (j = 0; j < classes; ++j) {
        //float prob = objectness * output[class_index + stride*j];
        float prob = output[class_index + stride*j];
        if (prob > conf_thresh) {
            return 1;
        }
    }
    return 0;
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

typedef struct train_yolo_args {
    layer l;
    network_state state;
    int b;

    float tot_iou;
    float tot_giou_loss;
    float tot_iou_loss;
    int count;
    int class_count;
} train_yolo_args;

void insert_organized(int **idx, float **loss, int value_idx, float value_loss, int nelem, int *number_valids)
{
//    printf("VALUE IDX: %d %d\n", value_idx, nelem);
//    printf("VALUE LOSS: %f\n", value_loss);

    *number_valids += (value_loss > 2*FLT_EPSILON);

    int i = 0;
    for (i = 0; i < nelem - 1; i++)
    {
      if ((value_loss > 2*FLT_EPSILON) && ((value_loss < *(*loss+i)) || (*(*loss+i) < 2*FLT_EPSILON))) //Organize only non-zero losses
        {
          float aux = *(*loss+i);
          *(*loss+i) = value_loss;
          value_loss = aux;

          int aux_int = *(*idx+i);
          *(*idx+i) = value_idx;
          value_idx = aux_int;
        }
    }

    if (nelem == 1){
        *loss = (float *)malloc(nelem * sizeof (float));
        *idx = (int *)malloc(nelem * sizeof (int));
    }
    else
    {
        *idx = realloc(*idx, nelem * sizeof (int));
        *loss = realloc(*loss, nelem * sizeof (float));
    }

    *(*idx+i) = value_idx;
    *(*loss+i) = value_loss;

//    printf("VALUE IDX OUT: %d %d\n", *(*idx+i), nelem);
//    printf("VALUE LOSS OUT: %f\n", value_loss);

}

void *process_batch(void* ptr)
{
    {
        srand(time(NULL));
        train_yolo_args *args = (train_yolo_args*)ptr;
        const layer l = args->l;
        network_state state = args->state;
        int b = args->b;

        int i, j, t, n;

        //printf(" b = %d \n", b, b);

        //float tot_iou = 0;
        float tot_giou = 0;
        float tot_diou = 0;
        float tot_ciou = 0;
        //float tot_iou_loss = 0;
        //float tot_giou_loss = 0;
        float tot_diou_loss = 0;
        float tot_ciou_loss = 0;
        float recall = 0;
        float recall75 = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        //int count = 0;
        //int class_count = 0;

        pid_t pid = syscall(__NR_gettid);


        typedef struct scale_augment {
            float x;
            float y;
            float dx;
            float dy;
            float sw;
            float sh;
            int flip;
        } scale_augment_t;

        scale_augment_t init_scales = {0};

        typedef struct objectnessij {
            float obj;
            int i;
            int j;
        } objectnessij_t;

        objectnessij_t* objects = (objectnessij_t*)xcalloc(1, sizeof(objectnessij_t));
        uint32_t number_of_t = 0;
        objects[number_of_t].obj = 0.0f;
        objects[number_of_t].i = 0;
        objects[number_of_t].j = 0;

        char* fname_base_i = (char*)malloc(100*sizeof(char));
        char* fname_hash_i = (char*)malloc(100*sizeof (char));

//        int* allowed_t = (int*)xcalloc(1,sizeof(int));
        int* sid_t = (int*)xcalloc(1,sizeof(int));
        uint32_t count_allowed = 0;

        for (t = 0; t < l.max_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
            if (!truth.x) break;

            int sid, lid, tid;
            tid = (int)state.truth[t * l.truth_size + b * l.truths + 5];

            sprintf(fname_hash_i, "hash/%d.txt", tid);
            if( access( fname_hash_i, F_OK ) == 0 ) {
                FILE* fhash = fopen(fname_hash_i, "r");
                if(fscanf(fhash, "%d %d %s\n", &sid, &lid, fname_base_i) != 3) {
                    printf("HASH FILE INCORRECT INIT %d %d %d %s\n", tid, sid, lid, fname_base_i);
                    continue;
//                    exit(0);
                }
                fclose(fhash);
            } else {
                printf("NO FILE HASH FOUND INIT %s\n", fname_hash_i);
                continue;
//                exit(0);
            }

            sid_t[count_allowed] = sid;
            count_allowed++;
            sid_t = (int*)xrealloc(sid_t, (count_allowed+1)*sizeof(int));

//            if (sid == 3) { // Only label
//                continue;
//            } else {
//                allowed_t[count_allowed] = t;
//                sid_t[count_allowed] = sid;
//                count_allowed++;

//                allowed_t = (int*)xrealloc(allowed_t, (count_allowed+1)*sizeof(int));
//                sid_t = (int*)xrealloc(sid_t, (count_allowed+1)*sizeof(int));
//            }
        }
        free(fname_base_i);
        free(fname_hash_i);

//        if(count_allowed > 0) {

            // RUNS ON EACH IMAGE CELL (j, i) and anchor n
            for (j = 0; j < l.h; ++j) { //printf("[%d] LAYER H->J: %d\n",pid,j);
                for (i = 0; i < l.w; ++i) { //printf("[%d] LAYER W->I: %d\n",pid,i);
                    for (n = 0; n < l.n; ++n) {
                        const int class_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4 + 1);
                        const int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4); // objecteness
                        const int box_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
                        const int stride = l.w * l.h;

//                        printf("NUMBER OF T: %d %d %d %d %d\n", number_of_t, j, i, n, l.n);

// >>>>>>>>>>>>>> COMMENTED
                        if (((j>0) || (i>0) || (n>0)) && (number_of_t>0)) {
                            float objectn = l.output[obj_index];
                            if (isnan(objectn) || isinf(objectn)) objectn = 0;
                            objectn *= l.output[class_index];

                            int found_ij = 0;
                            for (int a = 0; a < number_of_t; a++) {
                                if ((objects[a].i == i) && (objects[a].j == j)) { // Found same ij

                                    if(objects[a].obj > objectn) {
                                        found_ij = 1; // Found better ij
                                        break;
                                    }

                                    for (int b = a; b < number_of_t-1; b++) {
                                        objects[b].obj = objects[b+1].obj;
                                        objects[b].i = objects[b+1].i;
                                        objects[b].j = objects[b+1].j;
                                    }

                                    objects[number_of_t-1].obj = -1.0f; // Ensure the next if/for will work.

                                }
                            }


                            if ((found_ij == 0) && (objectn > objects[number_of_t-1].obj)) {
                                objects[number_of_t-1].obj = objectn;
                                objects[number_of_t-1].i = i;
                                objects[number_of_t-1].j = j;

//                                if (number_of_t > 1) {
                                    for (int a = number_of_t-2; a >= 0; a--) {
                                        if(objectn > objects[a].obj) {
                                            objects[a+1].obj = objects[a].obj;
                                            objects[a+1].i = objects[a].i;
                                            objects[a+1].j = objects[a].j;

                                            objects[a].obj = objectn;
                                            objects[a].i = i;
                                            objects[a].j = j;
                                        } else {
                                            break; // No need to keep searching
                                        }
                                    }
//                                }
                            }
                        }
// <<<<<<<<<<<<<<<< COMMENTED


                        box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.new_coords);
                        float best_match_iou = 0;
                        int best_match_t = 0;
                        float best_iou = 0;
                        int best_t = 0;
                        int best_sid = 0;

                        for (t = 0; t < l.max_boxes; ++t) {
//                        uint32_t iter = 0;
//                        while (iter < count_allowed) {
//                            t = allowed_t[iter];
//                            iter++;

                            box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
                            if (!truth.x) break;  // continue;
                            int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                            if (class_id >= l.classes || class_id < 0) {
                                printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                                printf("\n truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n", truth.x, truth.y, truth.w, truth.h, class_id);
                                if (check_mistakes) getchar();
                                continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value
                            }

                            float objectness = l.output[obj_index];
                            if (isnan(objectness) || isinf(objectness)) l.output[obj_index] = 0;
                            int class_id_match = compare_yolo_class(l.output, l.classes, class_index, l.w * l.h, objectness, class_id, 0.25f);

                            float iou = box_iou(pred, truth);
                            if (iou > best_match_iou && class_id_match == 1) {
                                best_match_iou = iou;
                                best_match_t = t;
                            }
                            if (iou > best_iou) {
                                best_iou = iou;
                                best_t = t;
                                best_sid = sid_t[t];
                            }

// >>>>>>>>>>>>>> COMMENTED
                            if ((j==0) && (i==0) && (n==0)) { // Create the array
                                objects[number_of_t].obj = t==0 ? l.output[obj_index]*l.output[class_index] : -1.0f;
                                objects[number_of_t].i = i;
                                objects[number_of_t].j = j;
                                number_of_t++;
                                objects = (objectnessij_t*)xrealloc(objects, (number_of_t+1)*sizeof (objectnessij_t));
                            }
// >>>>>>>>>>>>>> COMMENTED


//                            if (l.output[obj_index]*l.output[class_index] > objects[t].obj) {
//                                objects[t].obj = l.output[obj_index]*l.output[class_index];
//                                objects[t].i = i;
//                                objects[t].j = j;
//                            }

//                            if((uint32_t)t == number_of_t) {
//                                number_of_t++;
//                                objects = (objectnessij_t*)xrealloc(objects, (number_of_t+1)*sizeof (objectnessij_t));
//                                objects[number_of_t].obj = -1.0f;
//                                objects[number_of_t].i = i;
//                                objects[number_of_t].j = j;
//                            }
                        }

//                        if((best_t == 0) && (best_iou < 0.001f)) {
//                            continue;
//                        }

    //                    printf("OBJECTNESS SMOTH: %d MATCH: %f IGNORE: %f ADV: %d BEST: %f TRUTH: %f\n", l.objectness_smooth, best_match_iou, l.ignore_thresh, state.net.adversarial, best_iou, l.truth_thresh);

                        avg_anyobj += l.output[obj_index];
                        l.delta[obj_index] = l.obj_normalizer * (0 - l.output[obj_index]);
                        if (best_match_iou > l.ignore_thresh) {
                            if (l.objectness_smooth) {
                                const float delta_obj = l.obj_normalizer * (best_match_iou - l.output[obj_index]);
                                if (delta_obj > l.delta[obj_index]) l.delta[obj_index] = delta_obj;

                            }
                            else l.delta[obj_index] = 0;
                        }
                        else if (state.net.adversarial) {
                            int stride = l.w * l.h;
                            float scale = pred.w * pred.h;
                            if (scale > 0) scale = sqrt(scale);
                            l.delta[obj_index] = scale * l.obj_normalizer * (0 - l.output[obj_index]);
                            int cl_id;
                            int found_object = 0;
                            for (cl_id = 0; cl_id < l.classes; ++cl_id) {
                                if (l.output[class_index + stride * cl_id] * l.output[obj_index] > 0.25) {
                                    l.delta[class_index + stride * cl_id] = scale * (0 - l.output[class_index + stride * cl_id]);
                                    found_object = 1;
                                }
                            }
                            if (found_object) {
                                // don't use this loop for adversarial attack drawing
                                for (cl_id = 0; cl_id < l.classes; ++cl_id)
                                    if (l.output[class_index + stride * cl_id] * l.output[obj_index] < 0.25)
                                        l.delta[class_index + stride * cl_id] = scale * (1 - l.output[class_index + stride * cl_id]);

//                                l.delta[box_index + 0 * stride] += scale * (0 - l.output[box_index + 0 * stride]); // * (1-0.5f*(best_sid==4));
//                                l.delta[box_index + 1 * stride] += scale * (0 - l.output[box_index + 1 * stride]); // * (1-0.5f*(best_sid==4));
//                                l.delta[box_index + 2 * stride] += scale * (0 - l.output[box_index + 2 * stride]); // * (0.5f+0.5f*(best_sid>0));
//                                l.delta[box_index + 3 * stride] += scale * (0 - l.output[box_index + 3 * stride]); // * (0.5f+0.5f*(best_sid>0));
                                float alpha = 0.75f;
                                float beta = 0.5f;

                                if(best_sid < 3) {
                                    l.delta[box_index + 0 * stride] += scale * (0 - l.output[box_index + 0 * stride]);
                                    l.delta[box_index + 1 * stride] += scale * (0 - l.output[box_index + 1 * stride]);
                                } else if (best_sid==4) {
                                    l.delta[box_index + 0 * stride] += scale * (0 - l.output[box_index + 0 * stride]) * (alpha);
                                    l.delta[box_index + 1 * stride] += scale * (0 - l.output[box_index + 1 * stride]) * (alpha);
                                }
                                if(best_sid != 3) {
                                    l.delta[box_index + 2 * stride] += scale * (0 - l.output[box_index + 2 * stride]) * (alpha+(1-alpha)*(best_sid==0)); // 1 if fully annotated
                                    l.delta[box_index + 3 * stride] += scale * (0 - l.output[box_index + 3 * stride]) * (alpha+(1-alpha)*(best_sid==0));
                                }
                            }
                        }
//                        printf("\n\nTRUTH THRESHHHH %f\n\n", l.truth_thresh);
                        if (best_iou > l.truth_thresh) { // Probably never enters here l.truth_thresh = 1 by default
                            const float iou_multiplier = best_iou * best_iou;// (best_iou - l.truth_thresh) / (1.0 - l.truth_thresh);
                            if (l.objectness_smooth) l.delta[obj_index] = l.obj_normalizer * (iou_multiplier - l.output[obj_index]);
                            else l.delta[obj_index] = l.obj_normalizer * (1 - l.output[obj_index]);
                            //l.delta[obj_index] = l.obj_normalizer * (1 - l.output[obj_index]);

                            int class_id = state.truth[best_t * l.truth_size + b * l.truths + 4];
                            if (l.map) class_id = l.map[class_id];
                            delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, 0, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);
                            const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                            if (l.objectness_smooth) l.delta[class_index + stride * class_id] = class_multiplier * (iou_multiplier - l.output[class_index + stride * class_id]);
                            box truth = float_to_box_stride(state.truth + best_t * l.truth_size + b * l.truths, 1);
    //                        delta_yolo_box((int)(state.truth[box_index * l.truth_size + b * l.truths + 5]), 1, 1, truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                            delta_yolo_box((int)(state.truth[box_index * l.truth_size + b * l.truths + 5]), 1, 1, truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);

                            (*state.net.total_bbox)++;
                        }
                    }
                }
            }
//        }

////        free(fname_base_i);
////        free(fname_hash_i);
//        free(allowed_t);
        free(sid_t);

        float tx_ant=-1.0, ty_ant=-1.0, min_loss=0.0, acc_loss=0.0;
        int skip_loss = 0;

        unsigned int count = 0;
        int* used_boxes = (int*)xcalloc(1, sizeof(int));

//        used_boxes = (int*)xrealloc(used_boxes, (count + 1) * sizeof(int));

        int box_idx = -1;
        int min_box_idx = -1;

        int first = 1, t_ant = 0;
        int update_delta = 0;

        int cnt = 0;
        int final = 0;

        t = 0;

        int *idx_group = NULL;
        float *loss_group = NULL;

        int nelem_group = 1;
        int n_valid = 0;

        typedef struct box_group {
            int track_id;
            int *sample_id;
            float x;
            float y;
            float w;
            float h;
            int nelem;
            int *telem;
            float *loss;
        } box_group_t;

        // Struct to group representations over iterations
        box_group_t *group_boxes = NULL;
        int num_boxes = 0;

        int current_group = 0;
        int current_item = 0;

        int all_evaluated = 0;

        group_boxes = (box_group_t*)malloc(sizeof(box_group_t));
        group_boxes[0].nelem = 1;
        group_boxes[0].telem = (int *)malloc(group_boxes[0].nelem*sizeof(int));
        group_boxes[0].sample_id = (int *)malloc(group_boxes[0].nelem*sizeof(int));
        group_boxes[0].loss  = (float *)malloc(group_boxes[0].nelem*sizeof(float));



        printf("\nINIT PROCESS:\n");
//        for (t = 0; t < l.max_boxes; ++t) {
        while(t < l.max_boxes){

//            printf("INIT\n");
            box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);

            float best_iou = 0;
            int best_n = 0;
            box truth_shift = truth;

            if (!truth.x) // No more true boxes
            {
                break;

                if(group_boxes == NULL){ // If all evaluated already.
                    break;
                }

                all_evaluated = 1;
                current_group = 0;
            }

            else {

                if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                    char buff[256];
                    printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
                    sprintf(buff, "echo \"Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\" >> bad_label.list",
                        truth.x, truth.y, truth.w, truth.h);
                    system(buff);
                }
                int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                if (class_id >= l.classes || class_id < 0) continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value

                //  float best_iou = 0;
                //  int best_n = 0;
                i = (truth.x * l.w);
                j = (truth.y * l.h);
                //  box truth_shift = truth;
                truth_shift.x = truth_shift.y = 0;
                for (n = 0; n < l.total; ++n) { // Check best anchor
                    box pred = { 0 };
                    pred.w = l.biases[2 * n] / state.net.w;
                    pred.h = l.biases[2 * n + 1] / state.net.h;
//                    printf("[%d] PRED->W: %f  H: %f -- %d %d\n", pid, pred.w, pred.h, l.total, n);
                    float iou = box_iou(pred, truth_shift); // Compare only width-height
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_n = n;
                    }
                }
            }

//            int t_id = (*(int*)(&state.truth[t * l.truth_size + b * l.truths + 5]))%1000000;
//            t_id += ((int)(*(int*)(&state.truth[t * l.truth_size + b * l.truths + 5])/10000000))*10000000;

//            int s_id = (int)(*(int*)(&state.truth[t * l.truth_size + b * l.truths + 5])/1000000)%10;


            int s_id, img_id, label_id, t_id;
            char* fname_base = (char*)malloc(100*sizeof(char));

            printf("(%x)HERE %d %f\n", pid, (int)state.truth[t * l.truth_size + b * l.truths + 5], state.truth[t * l.truth_size + b * l.truths + 5]);

//            t_id = (*(int*)(&state.truth[t * l.truth_size + b * l.truths + 5]));
            t_id = (int)state.truth[t * l.truth_size + b * l.truths + 5];

////            const int truth_in_index = t * l.truth_size + b * l.truths + 5;
////            const int t_id = state.truth[truth_in_index];
//            printf("(%x)HERE %d\n", pid, t_id);

//            if( access( "current_batch.txt", F_OK ) == 0 ) {
//                FILE* fbatch = fopen("current_batch.txt", "r");
//                printf("(%x)HERE2 %d %d\n", pid, t_id, img_id);

//                while(fscanf(fbatch, "%d %d %d %s\n", &img_id, &s_id, &label_id, fname_base)==4){
////                    fscanf(fbatch, "%d %d %d %s\n", &img_id, &s_id, &label_id, fname_base);

//                    printf("(%x)HERE3 %d %d\n", pid, t_id, img_id);
//                    if(t_id == img_id) {
//                        break;
//                    }
//                }

//                fclose(fbatch);
//            }

            char* fname_hash;
            fname_hash = (char*)malloc(100*sizeof (char));

            sprintf(fname_hash, "hash/%d.txt", t_id);
            if( access( fname_hash, F_OK ) == 0 ) {
                FILE* fhash = fopen(fname_hash, "r");
                if(fscanf(fhash, "%d %d %s\n", &s_id, &label_id, fname_base) != 3) {
                    printf("HASH FILE INCORRECT %d %d %d %s\n", t_id, s_id, label_id, fname_base);
                    t++;
                    continue;
//                    exit(0);
                }
                fclose(fhash);
            } else {
                printf("NO FILE HASH FOUND %s\n", fname_hash);
//                exit(0);
                t++;
                continue;
            }

            free(fname_hash);

            printf("(%x)TRACK_ID %d %d\n", pid, t_id, s_id);
            free(fname_base);

            num_boxes=1;
//                    group_boxes = (box_group_t*)malloc(num_boxes*sizeof(box_group_t));
            group_boxes[0].track_id = t_id;
            group_boxes[0].x = truth.x;
            group_boxes[0].y = truth.y;
            group_boxes[0].w = truth.w;
            group_boxes[0].h = truth.h;
            group_boxes[0].nelem = 1;

//                *(group_boxes[0].telem) = t;
            group_boxes[0].telem[0] = t;
            group_boxes[0].sample_id[0] = s_id;
            current_group = 0;
            current_item = 0;

            printf("(%x)X: %f Y: %f W: %f H: %f\n",  pid, truth.x,  truth.y,  truth.w,  truth.h);

            min_box_idx = group_boxes[current_group].telem[0];
            printf("(%x)[ISOLATED BOX: T: %d G: %d E: %d] %d %f\n", pid, num_boxes, current_group, 0, group_boxes[current_group].telem[0], group_boxes[current_group].loss[0]);

            update_delta = 1;

            if(s_id == 3) {
                printf("\n\n(%x)CHANGING I J to %d %d %f %d %d\n\n", pid, objects[t].i, objects[t].j, objects[t].obj, l.w, l.h);

                for (uint32_t a=0;a<number_of_t;a++) {
                    printf("(%x)CONFS %f %d %d %d %d\n", pid, objects[a].obj, (int)state.truth[a * l.truth_size + b * l.truths + 5], objects[a].i, objects[a].j, number_of_t);
                }
                i = objects[t].i;
                j = objects[t].j;
            }

            {
//            if (all_evaluated == 0) { // Analysing all boxes first

//            }
//            else {
//                if (update_delta == 0){ // Only calculating losses before

//                    float sum_loss = 0.0f;
//                    float scale_zeros = 1.0f;
//                    int nelem_valid = 0;
//                    float zero_loss_prob = 0.1f;



////                    if(group_boxes[current_group].nelem > 1) {
//////                      /*  printf("ORGANIZED LIST\n");
////                        for (int z = 0; z<group_boxes[current_group].nelem; z++)
////                        {
////                            if(group_boxes[current_group].loss[z] < 2*FLT_EPSILON) {
////                                scale_zeros -= zero_loss_prob; //Give a % for non-overlapping solutions
////                            }
////                            else {
////                                sum_loss += 1.0f/group_boxes[current_group].loss[z];
////                                nelem_valid++;
////                            }
//////                            printf("(%x)X: %f Y: %f W: %f H: %f\n",  pid, truth.x,  truth.y,  truth.w,  truth.h);

////                            printf("(%x)[ORGANIZED LIST: T: %d G: %d E: %d] %d %f\n", pid, num_boxes, current_group, z, group_boxes[current_group].telem[z], group_boxes[current_group].loss[z]);
////                        }

////                        float scale_prob = 1.0f/(group_boxes[current_group].nelem);

////                        if(sum_loss > 2*FLT_EPSILON) {
////                            scale_prob = (1.0/sum_loss);
////                        }

////                        float acc_prob = 0.0f;
////                        float prob_chosen = ((float)rand() / RAND_MAX * (1.0f - 0.0f)) + 0.0f;

////    //                            printf("PROB: %f SCALE: %f ZEROS_SC: %f\n", prob_chosen, scale_prob, scale_zeros);

////                        for (int z = 0; z<group_boxes[current_group].nelem; z++)
////                        {
//////                            min_box_idx = group_boxes[current_group].telem[0];
//////                            current_item = z;
//////                            break;

////                            if(group_boxes[current_group].loss[z] > 2*FLT_EPSILON){
////                                // (loss_elem/sum_loss*zero_scale + zero_prob) / (1+nelem*zero_prob - (1-zero_scale))
////                                acc_prob += (((1.0f/group_boxes[current_group].loss[z])*scale_prob) * scale_zeros + zero_loss_prob)/(1 + group_boxes[current_group].nelem*zero_loss_prob - (1-scale_zeros));
////                            }
////                            else {
////                                if (sum_loss < 2*FLT_EPSILON) { // Only 0 losses
////                                    acc_prob += scale_prob;
////                                }
////                                else {
////                                    acc_prob += zero_loss_prob/(1 + group_boxes[current_group].nelem*zero_loss_prob - (1-scale_zeros));
////                                }
////                            }

////    //                                printf("PROB: %f LOSS: %f ID: %d\n", acc_prob, group_boxes[current_group].loss[z], group_boxes[current_group].telem[z]);

////                            if(prob_chosen <= acc_prob) {
////                                min_box_idx = group_boxes[current_group].telem[z];
////                                current_item = z;
////                                min_box_idx = group_boxes[current_group].telem[0];
////                                current_item = 0;

//////                                int sid = group_boxes[current_group].sample_id[z];
////                                int sid = group_boxes[current_group].sample_id[0];
////                                //                                    printf("PROB: %f ACC: %f IDX_CHOSEN: %d\n", prob_chosen, acc_prob, min_box_idx);

////                                if ((group_boxes[current_group].loss[current_item] < 0.25f) && (group_boxes[current_group].loss[current_item] > 0.001f)){
//////                                if(1){

////                                    int img_id = (*(int*)(&state.truth[min_box_idx * l.truth_size + b * l.truths + 5]))%1000000;
////                                    int label_id = ((*(int*)(&state.truth[min_box_idx * l.truth_size + b * l.truths + 5]))/10000000);

////                                    char* fname_init;
////                                    fname_init = (char*)malloc(100*sizeof (char));
////                                    sprintf(fname_init, "/home/fazevedo/Desktop/PhD/COCOPerson/labels/train/%012d_init.txt", img_id);
////                                    printf("\n\n\nNAME OF INIT FILE %s %d\n", fname_init, label_id);

////                                    if( access( fname_init, F_OK ) == 0 ) {
////                                        FILE* finit = fopen(fname_init, "r");

////                                        int i = 0;

////                                        float x, y, h, w;
////                                        int id = 0;

////                                        while(fscanf(finit, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5) {
////                                            if (i != sid*label_id){ // Verify if it is sample ID needed.
////                                                i++;
////                                                continue;
////                                            }

////                                            char* filename;
////                                            filename = (char*)malloc(100*sizeof (char));
////                                            sprintf(filename, "/home/fazevedo/Desktop/PhD/COCOPerson/labels/train/%012d_iteration.txt", img_id);
////                                            printf("\n\n\nNAME OF ITERATION FILE %s\n", filename);

////                                            float ax, ay, ah, aw;
////                                            int aid;
////                                            int found = 0;
////                                            char* buffer;
////                                            buffer = (char*)malloc(1*sizeof (char));
////                                            int n_char = 0, line_size=42;

////                                            if( access( filename, F_OK ) == 0 ) {
////                                                FILE* fiter = fopen(filename, "r");

////                                                while(fscanf(fiter, "%d %f %f %f %f", &aid, &ax, &ay, &aw, &ah) == 5) {
////                                                    n_char+=line_size; // All line plus \n
////                                                    buffer = realloc(buffer, n_char+1);

////                                                    if ((aid == id) &&
////                                                        (fabs(x-ax) < 2*FLT_EPSILON) &&
////                                                        (fabs(y-ay) < 2*FLT_EPSILON) ){
////                                                        found = 1;
////                                                        // REPLACE LINE
////                                                        sprintf(buffer+(n_char-line_size), "%d %.7f %.7f %.7f %.7f\n", id, x, y, w, h);
////                                                    }
////                                                    else {
////                                                        sprintf(buffer+(n_char-line_size), "%d %.7f %.7f %.7f %.7f\n", aid, ax, ay, aw, ah);
////                                                    }

////                                                };
////                                                fclose(fiter);

////                                                if (found == 0){
////                                                    // ADD Line
////                                                    n_char+=line_size; // All line plus \n
////                                                    buffer = realloc(buffer, n_char+1);
////                                                    sprintf(buffer+(n_char-line_size), "%d %.7f %.7f %.7f %.7f\n", id, x, y, w, h);
////            //                                        fprintf(fw, "%d %.7f %.7f %.7f %.7f\n", sid, sx, sy, sw, sh);
////                                                }

////                                                FILE* fw = fopen(filename, "w");
////                                                buffer[n_char] = '\0';
////                                                fprintf(fw, "%s", buffer);
////                                                fclose(fw);
////                                                free(buffer);
////                                            } else {
////                                                // file doesn't exist
////                                                printf("FILE ITER DOES NOT EXIST\n");
////                                                FILE* fw = fopen(filename, "w");
////                                                fprintf(fw, "%d %.7f %.7f %.7f %.7f\n", id, x, y, w, h);
////                                                fclose(fw);
////                                            }

////                                            free(filename);

////                                            break;
////                                        }
////                                        fclose(finit);
////                                    }
////                                    else {
////                                        printf("NO INIT FILE\n");
////                                    }

////                                    free(fname_init);
////                                } //<<<< If Loss small
////                                break;
////                            }
////                        }
////                    }
////                    else {
//                        current_item = 0;
//                        min_box_idx = group_boxes[current_group].telem[0];

////                        printf("PID: %x\n", pid);
//                        printf("(%x)[ISOLATED BOX: T: %d G: %d E: %d] %d %f\n", pid, num_boxes, current_group, 0, group_boxes[current_group].telem[0], group_boxes[current_group].loss[0]);

//    //                    printf("PROB: %f ACC: %f IDX_CHOSEN: %d\n", prob_chosen, acc_prob, min_box_idx);

////                        box prediction = get_yolo_box(l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.new_coords);

////                        int img_id = (*(int*)(&state.truth[min_box_idx * l.truth_size + b * l.truths + 5]))%1000000;
////                        int label_id = ((*(int*)(&state.truth[min_box_idx * l.truth_size + b * l.truths + 5]))/10000000);

////                        char* fname_init;
////                        fname_init = (char*)malloc(100*sizeof (char));
////                        sprintf(fname_init, "/home/fazevedo/Desktop/PhD/COCOPerson/labels/train/%012d_init.txt", img_id);
////                        printf("\n\n\n(%x)NAME OF INIT FILE %s %d\n", pid, fname_init, label_id);

////                        char* fname_scale;
////                        fname_scale = (char*)malloc(100*sizeof (char));
////                        sprintf(fname_scale, "/home/fazevedo/Desktop/PhD/COCOPerson/labels/train/%012d_scale.txt", img_id);
////                        printf("(%x)NAME OF SCALE FILE %s %d\n", pid, fname_scale, label_id);

////                        if( access( fname_init, F_OK ) == 0 ) {
////                            FILE* finit = fopen(fname_init, "r");

////                            int i = 0;

////                            float h, w;
////                            int id = 0;

////                            while(fscanf(finit, "%d %f %f %f %f", &id, &init_scales.x, &init_scales.y, &w, &h) == 5) {
////                                if (i != label_id){ // Verify if it is sample ID needed.
////                                    i++;
////                                    continue;
////                                }
////                                break;
////                            }
////                            fclose(finit);

////                            if( access( fname_scale, F_OK ) == 0 ) {
////                                FILE* fscale = fopen(fname_scale, "r");

////                                if(fscanf(fscale, "%f %f %f %f %d", &init_scales.dx, &init_scales.dy, &init_scales.sw, &init_scales.sh, &init_scales.flip) == 5) {
////                                    printf("(%x)SCALES: %f %f %f %f %f %f %d\n", pid, init_scales.x, init_scales.y, init_scales.dx, init_scales.dy, init_scales.sw, init_scales.sh, init_scales.flip);

////                                    float x_rec = (fabs(init_scales.flip - group_boxes[current_group].x) + init_scales.dx)/init_scales.sw;
////                                    w = 0;
////                                    w += 2*fabs(x_rec-init_scales.x);

////                                    float y_rec = (group_boxes[current_group].y + init_scales.dy)/init_scales.sh;
////                                    h = 0;
////                                    h += 2*fabs(y_rec-init_scales.y);

////                                    printf("(%x)RECOVER: %f %f %f %f\n", pid, x_rec, y_rec, w, h);
////                                }

////                                fclose(fscale);
////                            }
////                        }

////                        free(fname_init);
////                        free(fname_scale);


////                    }

////                    t = min_box_idx;
//                    update_delta = 1;

//                    //// VERIFY THIS!!!
////                    current_group++; //To iterate next
//                    continue;

//                }
//            }
            }

            if(update_delta)
                printf("(%x)EVALUATING G: %d E: %d\n", pid, current_group, t);

//            printf("NUMBER of GROUPS: %d\n", num_boxes);

            acc_loss = 0;

            int mask_n = int_index(l.mask, best_n, l.n);

//            int s_id = (int)(*(int*)(&state.truth[t * l.truth_size + b * l.truths + 5])/1000000)%10;

            if ((s_id == 1) || (s_id == 3)) {
                float best_conf_value = 0.0f;
                for (int z=0; z<l.total; z++) {
                    int test_mask = int_index(l.mask, z, l.n);
                    if (test_mask > -1) {
                        float conf = l.output[entry_index(l, b, test_mask * l.w * l.h + j * l.w + i, 4)];
                        if (conf > best_conf_value) {
                            best_conf_value = conf;
                            best_n = z;
                            printf("(%x)BEST CONF: %f %d\n", pid, best_conf_value, z);
                        }
                    }
                }
                if (best_conf_value < 0.1f) {
                    mask_n = -1;
                } else {
                    mask_n = int_index(l.mask, best_n, l.n);
                }
            }

//            for (int z=0; z<l.total; z++) {
//                printf("(%x)CONF %d %d: %f\n", pid, z, int_index(l.mask, z, l.n), l.output[entry_index(l, b, z * l.w * l.h + j * l.w + i, 4)]);
//            }



//            printf("MASK: %d\n", l.mask[0]);
//            printf("IJ: %d %d\n", i,j);
            if (mask_n >= 0) { // Check if anchor is from the output evaluated (0-2 first, 3-5 medium, 6-8 last)
                int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                if (l.map) class_id = l.map[class_id];

                int box_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                int conf_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);

//                if (l.output[conf_index] > 0.25f) {
                if (1) {
    //                avg_obj += l.output[obj_index];
//                    printf("BOX PRED IDX: %d %f\n", box_index, l.output[conf_index]);
                    const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                    ious all_ious = delta_yolo_box((int)(state.truth[t * l.truth_size + b * l.truths + 5]), group_boxes[current_group].nelem, update_delta, truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
//                    ious all_ious = delta_yolo_box(update_delta*(group_boxes[current_group].nelem==1), truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);

    //                acc_loss += all_ious.ciou;
                    acc_loss += 1 - all_ious.iou;
//                    acc_loss += 1 - l.output[conf_index];
//                    printf("ACC LOSS: %f \n", acc_loss);
//                    if (update_delta && (l.output[conf_index] < 1.01f - 0.96f*(group_boxes[current_group].nelem>1)))
//                    const int track_id = *(int*)(&state.truth[t * l.truth_size + b * l.truths + 5]);
                    const int track_id = (int)(state.truth[t * l.truth_size + b * l.truths + 5]);
                    printf("(%x)IMG: %d LABEL: %f %f %f %f\n", pid, track_id, truth.x, truth.y, truth.w, truth.h);
                    if (update_delta)// && (group_boxes[current_group].nelem==1))
                    {

                        (*state.net.total_bbox)++;

                        const int truth_in_index = t * l.truth_size + b * l.truths + 5;
//                        const int track_id = state.truth[truth_in_index];
                        const int truth_out_index = b * l.n * l.w * l.h + mask_n * l.w * l.h + j * l.w + i;
                        l.labels[truth_out_index] = track_id;
                        l.class_ids[truth_out_index] = class_id;

//                        printf("IMG: %d LABEL: %f %f %f %f\n", track_id, truth.x, truth.y, truth.w, truth.h);

                        //printf(" track_id = %d, t = %d, b = %d, truth_in_index = %d, truth_out_index = %d \n", track_id, t, b, truth_in_index, truth_out_index);

                        // range is 0 <= 1
                        args->tot_iou += all_ious.iou;//*(group_boxes[current_group].nelem==1);
//                        args->tot_iou_loss += 1 - all_ious.iou;

                        printf("GROUP ELEMS: %d %d\n", current_group, group_boxes[current_group].nelem);
                        args->tot_iou_loss += (1 - all_ious.iou)/group_boxes[current_group].nelem;//*(group_boxes[current_group].nelem==1);

//                        args->tot_iou_loss += 1 - l.output[conf_index];
//                        args->tot_iou_loss += all_ious.ciou;

    //    //                acc_loss += all_ious.ciou;
    //    //                if (update_delta)
    //    //                {
    //                        if (FLT_MAX - args->tot_iou_loss > all_ious.ciou)
    //                            args->tot_iou_loss += all_ious.ciou;
    //                        else
    //                            args->tot_iou_loss = FLT_MAX;
    //    //                }

                        // range is -1 <= giou <= 1
                        tot_giou += all_ious.giou;
                        args->tot_giou_loss += 1 - all_ious.giou;

                        tot_diou += all_ious.diou;
                        tot_diou_loss += 1 - all_ious.diou;

                        tot_ciou += all_ious.ciou;
                        tot_ciou_loss += 1 - all_ious.ciou;

                        int obj_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
                        avg_obj += l.output[obj_index];
                        if (l.objectness_smooth) {
                            float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
                            if (l.delta[obj_index] == 0) l.delta[obj_index] = delta_obj;
                        }
                        else l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);

                        int class_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

                        //printf(" label: class_id = %d, truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", class_id, truth.x, truth.y, truth.w, truth.h);
                        //printf(" mask_n = %d, l.output[obj_index] = %f, l.output[class_index + class_id] = %f \n\n", mask_n, l.output[obj_index], l.output[class_index + class_id]);

                        ++(args->count);
                        ++(args->class_count);
                        if (all_ious.iou > .5) recall += 1;
                        if (all_ious.iou > .75) recall75 += 1;
                    }
                }
            }

            // iou_thresh
            for (n = 0; n < l.total; ++n) {
                int mask_n = int_index(l.mask, n, l.n);
                if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) {
                    box pred = { 0 };
                    pred.w = l.biases[2 * n] / state.net.w;
                    pred.h = l.biases[2 * n + 1] / state.net.h;
                    float iou = box_iou_kind(pred, truth_shift, l.iou_thresh_kind); // IOU, GIOU, MSE, DIOU, CIOU
                    // iou, n

                    if (iou > l.iou_thresh) {
                        int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                        if (l.map) class_id = l.map[class_id];

                        int box_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                        box_idx = box_index;
                        int cf_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        ious all_ious = delta_yolo_box((int)(state.truth[t * l.truth_size + b * l.truths + 5]), group_boxes[current_group].nelem, update_delta, truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);

                        //                        ious all_ious = delta_yolo_box(*(int*)(&state.truth[t * l.truth_size + b * l.truths + 5]), group_boxes[current_group].nelem, update_delta, truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
//                        ious all_ious = delta_yolo_box(update_delta*(group_boxes[current_group].nelem==1), truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);

//                        acc_loss += all_ious.ciou;
                        acc_loss += 1 - all_ious.iou;
//                        acc_loss += 1 - l.output[cf_index];
                        if(update_delta)// && (group_boxes[current_group].nelem==1))
//                        if (update_delta && (l.output[cf_index] > 0.05f*(group_boxes[current_group].nelem>1)))
                        {
                            (*state.net.total_bbox)++;

                            // range is 0 <= 1
                            args->tot_iou += all_ious.iou;//*(group_boxes[current_group].nelem==1);
//                            args->tot_iou_loss += 1 - all_ious.iou;
                            printf("GROUP ELEMS: %d %d\n", current_group, group_boxes[current_group].nelem);
                            args->tot_iou_loss += (1 - all_ious.iou)/group_boxes[current_group].nelem;//*(group_boxes[current_group].nelem==1);


//                            args->tot_iou_loss += 1 - l.output[cf_index];
//                            args->tot_iou_loss += all_ious.ciou;

//    //                        acc_loss += all_ious.ciou;
//    //                        if(update_delta)
//    //                        {
//                                if (FLT_MAX - args->tot_iou_loss > all_ious.ciou)
//                                    args->tot_iou_loss += all_ious.ciou;
//                                else
//                                    args->tot_iou_loss = FLT_MAX;
//    //                        }
    //                        // range is -1 <= giou <= 1
                            tot_giou += all_ious.giou;
                            args->tot_giou_loss += 1 - all_ious.giou;

                            tot_diou += all_ious.diou;
                            tot_diou_loss += 1 - all_ious.diou;

                            tot_ciou += all_ious.ciou;
                            tot_ciou_loss += 1 - all_ious.ciou;

                            int obj_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
                            avg_obj += l.output[obj_index];
                            if (l.objectness_smooth) {
                                float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
                                if (l.delta[obj_index] == 0) l.delta[obj_index] = delta_obj;
                            }
                            else l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);

                            int class_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4 + 1);
                            delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

                            ++(args->count);
                            ++(args->class_count);
                            if (all_ious.iou > .5) recall += 1;
                            if (all_ious.iou > .75) recall75 += 1;
                        }
                    }
                }
            }

//            printf("CP1 %d %d %d\n", update_delta, current_group, num_boxes);

            if(update_delta == 1)
            {
                update_delta = 0;
                current_group++;

            }
            else{
//                printf("THIS ELEM_GROUP: %d %d %f\n", nelem_group, t, acc_loss);
//                insert_organized(&idx_group, &loss_group, t, (float)acc_loss, nelem_group, &n_valid);
//                printf("THIS ELEM_GROUP VAL: %d %d\n", (int)(nelem_group/2), idx_group[nelem_group-1]);

//                printf("CHOSEN out of %d: %d %f\n", nelem_group, *(idx_group + (int)(n_valid/2)), *(loss_group + (int)(n_valid/2)));
//                min_box_idx = *(idx_group + (int)(n_valid/2)); //Get always middle value for valid (loss>0)
//                min_box_idx = *(idx_group + (int)(nelem_group/2)); //Get always middle value
//                min_box_idx = *(idx_group); //Get always best value

                group_boxes[current_group].loss[current_item] = acc_loss;
//                printf("ACC LOSS: %f\n", acc_loss);

                // ORGANIZE LOSSES
                if (group_boxes[current_group].nelem > 1)
                {
                    for (int a = 0; a < group_boxes[current_group].nelem-1; a++) {
                        for (int b = a; b < group_boxes[current_group].nelem; ++b) {
                            if( ((group_boxes[current_group].loss[a] > group_boxes[current_group].loss[b]) && (group_boxes[current_group].loss[b] > 2*FLT_EPSILON)) ||
                                 (group_boxes[current_group].loss[a] < 2*FLT_EPSILON)){
                                float aux_loss = group_boxes[current_group].loss[a];
                                group_boxes[current_group].loss[a] = group_boxes[current_group].loss[b];
                                group_boxes[current_group].loss[b] = aux_loss;

                                int aux_t = group_boxes[current_group].telem[a];
                                group_boxes[current_group].telem[a] = group_boxes[current_group].telem[b];
                                group_boxes[current_group].telem[b] = aux_t;

                                int aux_s = group_boxes[current_group].sample_id[a];
                                group_boxes[current_group].sample_id[a] = group_boxes[current_group].sample_id[b];
                                group_boxes[current_group].sample_id[b] = aux_s;

                            }
                        }
                    }
                }

            }


            t++;

//            if(all_evaluated==0){
//                t++;
//            }
////            if(final==1)
////                break;
//            if((all_evaluated==1) && (current_group==num_boxes))
//                break;
        }

//        args->tot_iou_loss += min_loss;
//        used_boxes[count] = min_box_idx;

        //// ^^^ADD^^^

        if (l.iou_thresh < 1.0f) {
            // averages the deltas obtained by the function: delta_yolo_box()_accumulate
            for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w; ++i) {
                    for (n = 0; n < l.n; ++n) {
                        int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                        int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        const int stride = l.w*l.h;

                        if (l.delta[obj_index] != 0)
                            averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta);
                    }
                }
            }
        }

        free(used_boxes);

//        printf("FREE BOXES");
        for (int i = num_boxes-1; i >= 0; i--) {
            group_boxes[i].nelem = 0;
            free(group_boxes[i].loss);
            free(group_boxes[i].telem);
            free(group_boxes[i].sample_id);
        }
        free(group_boxes);
        num_boxes = 0;

        free(objects);

    }

    return 0;
}



void forward_yolo_layer(const layer l, network_state state)
{
    //int i, j, b, t, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
    int b, n;

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int bbox_index = entry_index(l, b, n*l.w*l.h, 0);
            if (l.new_coords) {
                //activate_array(l.output + bbox_index, 4 * l.w*l.h, LOGISTIC);    // x,y,w,h
            }
            else {
                activate_array(l.output + bbox_index, 2 * l.w*l.h, LOGISTIC);        // x,y,
                int obj_index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array(l.output + obj_index, (1 + l.classes)*l.w*l.h, LOGISTIC);
            }
            scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + bbox_index, 1);    // scale x,y
        }
    }
#endif

    // delta is zeroed
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!state.train) return;

    int i;
    for (i = 0; i < l.batch * l.w*l.h*l.n; ++i) l.labels[i] = -1;
    for (i = 0; i < l.batch * l.w*l.h*l.n; ++i) l.class_ids[i] = -1;
    //float avg_iou = 0;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;

    float avg_iou_loss = 0;


    int num_threads = l.batch;
    pthread_t* threads = (pthread_t*)calloc(num_threads, sizeof(pthread_t));

    struct train_yolo_args* yolo_args = (train_yolo_args*)xcalloc(l.batch, sizeof(struct train_yolo_args));

    for (b = 0; b < l.batch; b++)
    {
        yolo_args[b].l = l;
        yolo_args[b].state = state;
        yolo_args[b].b = b;

        yolo_args[b].tot_iou = 0;
        yolo_args[b].tot_iou_loss = 0;
        yolo_args[b].tot_giou_loss = 0;
        yolo_args[b].count = 0;
        yolo_args[b].class_count = 0;

        if (pthread_create(&threads[b], 0, process_batch, &(yolo_args[b]))) error("Thread creation failed");
    }

    for (b = 0; b < l.batch; b++)
    {
        pthread_join(threads[b], 0);

        tot_iou += yolo_args[b].tot_iou;
        tot_iou_loss += yolo_args[b].tot_iou_loss;
        tot_giou_loss += yolo_args[b].tot_giou_loss;
        count += yolo_args[b].count;
        class_count += yolo_args[b].class_count;
    }

    free(yolo_args);
    free(threads);

    // Search for an equidistant point from the distant boundaries of the local minimum
    int iteration_num = get_current_iteration(state.net);
    const int start_point = state.net.max_batches * 3 / 4;
    //printf(" equidistant_point ep = %d, it = %d \n", state.net.equidistant_point, iteration_num);

    if ((state.net.badlabels_rejection_percentage && start_point < iteration_num) ||
        (state.net.num_sigmas_reject_badlabels && start_point < iteration_num) ||
        (state.net.equidistant_point && state.net.equidistant_point < iteration_num))
    {
        const float progress_it = iteration_num - state.net.equidistant_point;
        const float progress = progress_it / (state.net.max_batches - state.net.equidistant_point);
        float ep_loss_threshold = (*state.net.delta_rolling_avg) * progress * 1.4;

        float cur_max = 0;
        float cur_avg = 0;
        float counter = 0;
        for (i = 0; i < l.batch * l.outputs; ++i) {

            if (l.delta[i] != 0) {
                counter++;
                cur_avg += fabs(l.delta[i]);

                if (cur_max < fabs(l.delta[i]))
                    cur_max = fabs(l.delta[i]);
            }
        }

        cur_avg = cur_avg / counter;

        if (*state.net.delta_rolling_max == 0) *state.net.delta_rolling_max = cur_max;
        *state.net.delta_rolling_max = *state.net.delta_rolling_max * 0.99 + cur_max * 0.01;
        *state.net.delta_rolling_avg = *state.net.delta_rolling_avg * 0.99 + cur_avg * 0.01;

        // reject high loss to filter bad labels
        if (state.net.num_sigmas_reject_badlabels && start_point < iteration_num)
        {
            const float rolling_std = (*state.net.delta_rolling_std);
            const float rolling_max = (*state.net.delta_rolling_max);
            const float rolling_avg = (*state.net.delta_rolling_avg);
            const float progress_badlabels = (float)(iteration_num - start_point) / (start_point);

            float cur_std = 0;
            float counter = 0;
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (l.delta[i] != 0) {
                    counter++;
                    cur_std += pow(l.delta[i] - rolling_avg, 2);
                }
            }
            cur_std = sqrt(cur_std / counter);

            *state.net.delta_rolling_std = *state.net.delta_rolling_std * 0.99 + cur_std * 0.01;

            float final_badlebels_threshold = rolling_avg + rolling_std * state.net.num_sigmas_reject_badlabels;
            float badlabels_threshold = rolling_max - progress_badlabels * fabs(rolling_max - final_badlebels_threshold);
            badlabels_threshold = max_val_cmp(final_badlebels_threshold, badlabels_threshold);
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (fabs(l.delta[i]) > badlabels_threshold)
                    l.delta[i] = 0;
            }
            printf(" rolling_std = %f, rolling_max = %f, rolling_avg = %f \n", rolling_std, rolling_max, rolling_avg);
            printf(" badlabels loss_threshold = %f, start_it = %d, progress = %f \n", badlabels_threshold, start_point, progress_badlabels *100);

            ep_loss_threshold = min_val_cmp(final_badlebels_threshold, rolling_avg) * progress;
        }


        // reject some percent of the highest deltas to filter bad labels
        if (state.net.badlabels_rejection_percentage && start_point < iteration_num) {
            if (*state.net.badlabels_reject_threshold == 0)
                *state.net.badlabels_reject_threshold = *state.net.delta_rolling_max;

            printf(" badlabels_reject_threshold = %f \n", *state.net.badlabels_reject_threshold);

            const float num_deltas_per_anchor = (l.classes + 4 + 1);
            float counter_reject = 0;
            float counter_all = 0;
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (l.delta[i] != 0) {
                    counter_all++;
                    if (fabs(l.delta[i]) > (*state.net.badlabels_reject_threshold)) {
                        counter_reject++;
                        l.delta[i] = 0;
                    }
                }
            }
            float cur_percent = 100 * (counter_reject*num_deltas_per_anchor / counter_all);
            if (cur_percent > state.net.badlabels_rejection_percentage) {
                *state.net.badlabels_reject_threshold += 0.01;
                printf(" increase!!! \n");
            }
            else if (*state.net.badlabels_reject_threshold > 0.01) {
                *state.net.badlabels_reject_threshold -= 0.01;
                printf(" decrease!!! \n");
            }

            printf(" badlabels_reject_threshold = %f, cur_percent = %f, badlabels_rejection_percentage = %f, delta_rolling_max = %f \n",
                *state.net.badlabels_reject_threshold, cur_percent, state.net.badlabels_rejection_percentage, *state.net.delta_rolling_max);
        }


        // reject low loss to find equidistant point
        if (state.net.equidistant_point && state.net.equidistant_point < iteration_num) {
            printf(" equidistant_point loss_threshold = %f, start_it = %d, progress = %3.1f %% \n", ep_loss_threshold, state.net.equidistant_point, progress * 100);
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (fabs(l.delta[i]) < ep_loss_threshold)
                    l.delta[i] = 0;
            }
        }
    }

    if (count == 0) count = 1;
    if (class_count == 0) class_count = 1;

    if (l.show_details == 0) {
        float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
        *(l.cost) = loss;

        loss /= l.batch;

//        fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f), count: %d, total_loss = %f \n",
//            (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, state.index, tot_iou / count, count, loss);
    }
    else {
        // show detailed output

//        printf("SHOWING DETAILED\n\n\n");

        int stride = l.w*l.h;
        float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
        memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));


        int j, n;
        for (b = 0; b < l.batch; ++b) {
            for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w; ++i) {
                    for (n = 0; n < l.n; ++n) {
                        int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                        no_iou_loss_delta[index + 0 * stride] = 0;
                        no_iou_loss_delta[index + 1 * stride] = 0;
                        no_iou_loss_delta[index + 2 * stride] = 0;
                        no_iou_loss_delta[index + 3 * stride] = 0;
                    }
                }
            }
        }

        float classification_loss = l.obj_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
        free(no_iou_loss_delta);
        float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
        float iou_loss = loss - classification_loss;

        float avg_iou_loss = 0;
        *(l.cost) = loss;

        // gIOU loss + MSE (objectness) loss
        if (l.iou_loss == MSE) {
//        if (0) {
            *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);

//            ///////ADDED
////            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
//            avg_iou_loss += count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
////            tot_iou_loss = 0;
//            *(l.cost) = avg_iou_loss + classification_loss;
////            loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
////            fprintf(stderr, "\n\n\nLOSS HERE\n\n\n\n");
//            ///////
        }
        else {
            // Always compute classification loss both for iou + cls loss and for logging with mse loss
            // TODO: remove IOU loss fields before computing MSE on class
            //   probably split into two arrays
            if (l.iou_loss == GIOU) {
                avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
            }
            else {
                avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
            }
            *(l.cost) = avg_iou_loss + classification_loss;
        }


        loss /= l.batch;
        classification_loss /= l.batch;
        iou_loss /= l.batch;

        fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f), count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
            (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, state.index, tot_iou / count, count, classification_loss, iou_loss, loss);

        //fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
        //    (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
        //    classification_loss, iou_loss, loss);
    }
}

void backward_yolo_layer(const layer l, network_state state)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

// Converts output of the network to detection boxes
// w,h: image width,height
// netw,neth: network width,height
// relative: 1 (all callers seems to pass TRUE)
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    // network height (or width)
    int new_w = 0;
    // network height (or width)
    int new_h = 0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w / netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h / neth;
    for (i = 0; i < n; ++i) {

        box b = dets[i].bbox;
        // x = ( x - (deltaw/2)/netw ) / ratiow;
        //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
        b.x = (b.x - deltaw / 2. / netw) / ratiow;
        b.y = (b.y - deltah / 2. / neth) / ratioh;
        // scale to match rotation of incoming image
        b.w *= 1 / ratiow;
        b.h *= 1 / ratioh;

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }

        dets[i].bbox = b;
    }
}

/*
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
*/

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for(n = 0; n < l.n; ++n){
        for (i = 0; i < l.w*l.h; ++i) {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

int yolo_num_detections_batch(layer l, float thresh, int batch)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, batch, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    // This snippet below is not necessary
    // Need to comment it in order to batch processing >= 2 images
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.new_coords);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                if (l.embedding_output) {
                    get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, 0, dets[count].embeddings);
                }

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

int get_yolo_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch)
{
    int i,j,n;
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, batch, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, batch, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.new_coords);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                if (l.embedding_output) {
                    get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, batch, dets[count].embeddings);
                }

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, batch, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    if (l.embedding_output) {
        layer le = state.net.layers[l.embedding_layer_id];
        cuda_pull_array_async(le.output_gpu, l.embedding_output, le.batch*le.outputs);
    }

    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int bbox_index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            if (l.new_coords) {
                //activate_array_ongpu(l.output_gpu + bbox_index, 4 * l.w*l.h, LOGISTIC);    // x,y,w,h
            }
            else {
                activate_array_ongpu(l.output_gpu + bbox_index, 2 * l.w*l.h, LOGISTIC);    // x,y

                int obj_index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array_ongpu(l.output_gpu + obj_index, (1 + l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
            }
            if (l.scale_x_y != 1) scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + bbox_index, 1);      // scale x,y
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        if (l.mean_alpha && l.output_avg_gpu) mean_array_gpu(l.output_gpu, l.batch*l.outputs, l.mean_alpha, l.output_avg_gpu);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, state.net.loss_scale * l.delta_normalizer, l.delta_gpu, 1, state.delta, 1);
}
#endif
