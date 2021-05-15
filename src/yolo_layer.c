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

ious delta_yolo_box(int update_delta, box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss, int accumulate, float max_delta, int *rewritten_bbox, int new_coords)
{
    if (delta[index + 0 * stride] || delta[index + 1 * stride] || delta[index + 2 * stride] || delta[index + 3 * stride]) {
        (*rewritten_bbox)++;
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

    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);
    all_ious.iou = box_iou(pred, ltruth);
    all_ious.giou = box_giou(pred, ltruth);
    all_ious.diou = box_diou(pred, ltruth);
    all_ious.ciou = box_ciou(pred, ltruth);

//    printf("\n\n\n\n\nVALUES %f %f %f %f %f %f %f %f %f %d\n", truth.x, truth.y, pred.x, pred.y, truth.w, truth.h, pred.w, pred.h, all_ious.iou, iou_loss);

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
    if (iou_loss == MSE)    // old loss
//    if(0)
    {
        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2 * n]);
        float th = log(truth.h*h / biases[2 * n + 1]);

        if (new_coords) {
            //tx = (truth.x*lw - i + 0.5) / 2;
            //ty = (truth.y*lh - j + 0.5) / 2;
            tw = sqrt(truth.w*w / (4 * biases[2 * n]));
            th = sqrt(truth.h*h / (4 * biases[2 * n + 1]));
        }

        //printf(" tx = %f, ty = %f, tw = %f, th = %f \n", tx, ty, tw, th);
        //printf(" x = %f, y = %f, w = %f, h = %f \n", x[index + 0 * stride], x[index + 1 * stride], x[index + 2 * stride], x[index + 3 * stride]);

        // accumulate delta
        delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;
        delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;
        delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;
        delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;
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
        delta[index + 0 * stride] += dx;
        delta[index + 1 * stride] += dy;
        delta[index + 2 * stride] += dw;
        delta[index + 3 * stride] += dh;
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

        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    const int class_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4 + 1);
                    const int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
                    const int box_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
                    const int stride = l.w * l.h;
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.new_coords);
                    float best_match_iou = 0;
                    int best_match_t = 0;
                    float best_iou = 0;
                    int best_t = 0;
                    for (t = 0; t < l.max_boxes; ++t) {
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
                        }
                    }

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

                            l.delta[box_index + 0 * stride] += scale * (0 - l.output[box_index + 0 * stride]);
                            l.delta[box_index + 1 * stride] += scale * (0 - l.output[box_index + 1 * stride]);
                            l.delta[box_index + 2 * stride] += scale * (0 - l.output[box_index + 2 * stride]);
                            l.delta[box_index + 3 * stride] += scale * (0 - l.output[box_index + 3 * stride]);
                        }
                    }
                    if (best_iou > l.truth_thresh) {
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
                        delta_yolo_box(1, truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                        (*state.net.total_bbox)++;
                    }
                }
            }
        }

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
            float x;
            float y;
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

//        for (t = 0; t < l.max_boxes; ++t) {
        while(t < l.max_boxes){

//            printf("INIT\n");
            box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);

            float best_iou = 0;
            int best_n = 0;
            box truth_shift = truth;

            if (!truth.x)
            {
//               t_ant = t;
//               t = min_box_idx;
//               final = 1;
//               update_delta = 1;
//               if(t < 0)
//                   break;
//               continue;
                if(group_boxes == NULL){
                    break;
                }

                all_evaluated = 1;
                current_group = 0;
            }//break;  // continue;

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

//                float best_iou = 0;
//                int best_n = 0;
                i = (truth.x * l.w);
                j = (truth.y * l.h);
//                box truth_shift = truth;
                truth_shift.x = truth_shift.y = 0;
                for (n = 0; n < l.total; ++n) {
                    box pred = { 0 };
                    pred.w = l.biases[2 * n] / state.net.w;
                    pred.h = l.biases[2 * n + 1] / state.net.h;
                    float iou = box_iou(pred, truth_shift);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_n = n;
                    }
                }
            }

            if (all_evaluated == 0) {
//                printf("BOX: %d %f %f %f %f\n", t, truth.x, truth.y, truth.w, truth.h);

                if (group_boxes == NULL) {
                    num_boxes++;
                    group_boxes = (box_group_t*)malloc(num_boxes*sizeof(box_group_t));
                    group_boxes[0].x = truth.x;
                    group_boxes[0].y = truth.y;
                    group_boxes[0].nelem = 1;
                    group_boxes[0].telem = (int *)malloc(group_boxes[0].nelem*sizeof(int));
                    group_boxes[0].loss  = (float *)malloc(group_boxes[0].nelem*sizeof(float));
    //                *(group_boxes[0].telem) = t;
                    group_boxes[0].telem[0] = t;
                    current_group = 0;
                    current_item = 0;
                } else {
                    int i = 0;
                    for (i=0; i < num_boxes; i++)
                    {
                        if ((fabs(group_boxes[i].x-truth.x)>2*FLT_EPSILON) || (fabs(group_boxes[i].y-truth.y)>2*FLT_EPSILON)) {
                            continue;
                        }

                        group_boxes[i].nelem++;
                        group_boxes[i].telem = (int *)realloc(group_boxes[i].telem,  group_boxes[i].nelem*sizeof(int));
                        group_boxes[i].loss  = (float *)realloc(group_boxes[i].loss, group_boxes[i].nelem*sizeof(float));
                        group_boxes[i].telem[group_boxes[i].nelem-1] = t;

                        current_group = i;
                        current_item = group_boxes[i].nelem-1;
                        break;
                    }

                    if (i==num_boxes) //No matching found
                    {
                        num_boxes++;
                        group_boxes = (box_group_t*)realloc(group_boxes, num_boxes*sizeof(box_group_t));
                        group_boxes[i].x = truth.x;
                        group_boxes[i].y = truth.y;
                        group_boxes[i].nelem = 1;
                        group_boxes[i].telem = (int *)malloc(group_boxes[i].nelem*sizeof(int));
                        group_boxes[i].loss  = (float *)malloc(group_boxes[i].nelem*sizeof(float));
    //                    *(group_boxes[num_boxes-1].telem) = t;
                        group_boxes[i].telem[group_boxes[i].nelem-1] = t;

                        current_group = i;
                        current_item = group_boxes[i].nelem-1;
                    }
                }
            }
            else {
                if (update_delta == 0){

                    float sum_loss = 0.0f;
                    float scale_zeros = 1.0f;
                    int nelem_valid = 0;
                    float zero_loss_prob = 0.1f;

                    if(group_boxes[current_group].nelem > 1) {
//                         printf("ORGANIZED LIST\n");
                        for (int z = 0; z<group_boxes[current_group].nelem; z++)
                        {
                            if(group_boxes[current_group].loss[z] < 2*FLT_EPSILON) {
                                scale_zeros -= zero_loss_prob; //Give a % for non-overlapping solutions
                            }
                            else {
                                sum_loss += 1.0f/group_boxes[current_group].loss[z];
                                nelem_valid++;
                            }

//                            printf("ORGANIZED LIST: %d %f\n", group_boxes[current_group].telem[z], group_boxes[current_group].loss[z]);
                        }

                        float scale_prob = 1.0f/(group_boxes[current_group].nelem);

                        if(sum_loss > 2*FLT_EPSILON) {
                            scale_prob = (1.0/sum_loss);
                        }

                        float acc_prob = 0.0f;
                        float prob_chosen = ((float)rand() / RAND_MAX * (1.0f - 0.0f)) + 0.0f;

    //                            printf("PROB: %f SCALE: %f ZEROS_SC: %f\n", prob_chosen, scale_prob, scale_zeros);

                        for (int z = 0; z<group_boxes[current_group].nelem; z++)
                        {
                            if(group_boxes[current_group].loss[z] > 2*FLT_EPSILON){
                                acc_prob += (((1.0f/group_boxes[current_group].loss[z])*scale_prob) * scale_zeros + zero_loss_prob)/(1 + group_boxes[current_group].nelem*zero_loss_prob - (1-scale_zeros));
                            }
                            else {
                                if (sum_loss < 2*FLT_EPSILON) {
                                    acc_prob += scale_prob;
                                }
                                else {
                                    acc_prob += zero_loss_prob/(1 + group_boxes[current_group].nelem*zero_loss_prob - (1-scale_zeros));
                                }
                            }

    //                                printf("PROB: %f LOSS: %f ID: %d\n", acc_prob, group_boxes[current_group].loss[z], group_boxes[current_group].telem[z]);

                            if(prob_chosen <= acc_prob) {
                                min_box_idx = group_boxes[current_group].telem[z];
                                current_item = z;
    //                                    printf("PROB: %f ACC: %f IDX_CHOSEN: %d\n", prob_chosen, acc_prob, min_box_idx);
                                break;
                            }
                        }
                    }
                    else {
                        current_item = 0;
                        min_box_idx = group_boxes[current_group].telem[0];
    //                    printf("PROB: %f ACC: %f IDX_CHOSEN: %d\n", prob_chosen, acc_prob, min_box_idx);
                    }

                    t = min_box_idx;
                    update_delta = 1;

                    current_group++; //To iterate next
                    continue;

                }
            }

//            printf("NUMBER of GROUPS: %d\n", num_boxes);

            acc_loss = 0;
//            if ((fabs(tx_ant-truth.x)>2*FLT_EPSILON) || (fabs(ty_ant-truth.y)>2*FLT_EPSILON))
            if (0)
            {

                if ((min_box_idx > -1) && (t_ant < t)) // Only if comes from no repetition
                {

                    //////// vvvv PROBABILITY

                    if (nelem_group > 1)
                    {
                        // Select min_box_idx based on 1/loss probability
                        float sum_loss = 0.0f;
                        float scale_zeros = 1.0f;
                        int nelem_valid = 0;
                        float zero_loss_prob = 0.1f;

                        for (int z = 0; z<nelem_group; z++)
                        {
//                            printf("ORGANIZED LIST: %d %f\n", idx_group[z], loss_group[z]);
                            if(loss_group[z] < 2*FLT_EPSILON) {
                                scale_zeros -= zero_loss_prob; //Give a % for non-overlapping solutions
                            }
                            else {
                                sum_loss += 1.0f/loss_group[z];
                                nelem_valid++;
                            }
                        }

                        float scale_prob = 1.0f/(nelem_group);

                        if(sum_loss > 2*FLT_EPSILON) {
                            scale_prob = (1.0/sum_loss);
                        }

                        float acc_prob = 0.0f;
                        float prob_chosen = ((float)rand() / RAND_MAX * (1.0f - 0.0f)) + 0.0f;

//                        printf("PROB: %f SCALE: %f ZEROS_SC: %f\n", prob_chosen, scale_prob, scale_zeros);

                        for (int z = 0; z<nelem_group; z++)
                        {
                            if(loss_group[z] > 2*FLT_EPSILON){
                                acc_prob += (((1.0f/loss_group[z])*scale_prob) * scale_zeros + zero_loss_prob)/(1 + nelem_group*zero_loss_prob - (1-scale_zeros));
                            }
                            else {
                                if (sum_loss < 2*FLT_EPSILON) {
                                    acc_prob += scale_prob;
                                }
                                else {
                                    acc_prob += zero_loss_prob/(1 + nelem_group*zero_loss_prob - (1-scale_zeros));
                                }
                            }

//                            printf("PROB: %f SCALE: %f ZEROS_SC: %f\n", acc_prob);

                            if(prob_chosen <= acc_prob) {
                                min_box_idx = idx_group[z];
//                                printf("PROB: %f ACC: %f IDX_CHOSEN: %d\n", prob_chosen, acc_prob, min_box_idx);
                                break;
                            }
                        }
                    }
                    else {
                        min_box_idx = idx_group[0];
                    }

                    //////// ^^^^ PROBABILITY

                    t_ant = t;
                    t = min_box_idx;
                    update_delta = 1;

//                    printf("CONTINUE SAME BB %d %d %d\n", cnt, t, t_ant);

                    continue;
                }
//                {
//                    used_boxes[count] = min_box_idx;
//                    count++;
//                    min_box_idx = -1;
//                    used_boxes = (int*)xrealloc(used_boxes, (count) * sizeof(int));

//                }

//                if (skip_loss==1)
//                {
//                    skip_loss = 0;
//                }
                tx_ant = truth.x;
                ty_ant = truth.y;
//                args->tot_iou_loss += min_loss;
                min_loss=FLT_MAX;
                min_box_idx = t;
                cnt = 1;


                if (idx_group != NULL)
                {
//                    for (int z = 0; z<nelem_group-1; z++)
//                        printf("ORGANIZED LIST: %d %f\n", idx_group[z], loss_group[z]);

//                    if (nelem_group > 2)
//                    {
//                        // Select min_box_idx based on 1/loss probability
//                        float sum_loss = 0.0f;
//                        float scale_zeros = 1.0f;
//                        for (int z = 0; z<nelem_group-1; z++)
//                        {
//                            printf("ORGANIZED LIST: %d %f\n", idx_group[z], loss_group[z]);
//                            if(loss_group[z] < 2*FLT_EPSILON) {
//                                scale_zeros -= 0.05f; //Give 5% for non-overlapping solutions
//                            }
//                            else {
//                                sum_loss += loss_group[z];
//                            }
//                        }

//                        float scale_prob = 1.0f/(nelem_group-1);

//                        if(sum_loss > 2*FLT_EPSILON) {
//                            scale_prob = 1/sum_loss;
//                        }

//                        float acc_prob = 0.0f;
//                        float prob_chosen = ((float)rand() / RAND_MAX * (1.0f - 0.0f)) + 0.0f;

//                        for (int z = 0; z<nelem_group-1; z++)
//                        {
//                            if(loss_group[z] > 2*FLT_EPSILON){
//                                acc_prob += (1 - loss_group[z]*scale_prob) * scale_zeros;
//                            }
//                            else {
//                                if (sum_loss < 2*FLT_EPSILON) {
//                                    acc_prob += scale_prob;
//                                }
//                                else {
//                                    acc_prob += 0.05;
//                                }
//                            }

//                            if(prob_chosen <= acc_prob) {
//                                min_box_idx = idx_group[z];
//                                break;
//                            }
//                        }
//                    }
//                    else {
//                        min_box_idx = idx_group[0];
//                    }

                    free(idx_group);
                    free(loss_group);
                }

                nelem_group = 1;
                n_valid = 0;

            }
//            else {
//                nelem_group++;
//                cnt++;
////                printf("SAME BB %d %d %d\n", cnt, t, t_ant);
//                skip_loss = 1;
//            }
            // ^^^ COMMENTED

            int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                if (l.map) class_id = l.map[class_id];

                int box_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                ious all_ious = delta_yolo_box(update_delta, truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);

//                acc_loss += all_ious.ciou;
                acc_loss += 1 - all_ious.iou;
                if (update_delta)
                {

                    (*state.net.total_bbox)++;

                    const int truth_in_index = t * l.truth_size + b * l.truths + 5;
                    const int track_id = state.truth[truth_in_index];
                    const int truth_out_index = b * l.n * l.w * l.h + mask_n * l.w * l.h + j * l.w + i;
                    l.labels[truth_out_index] = track_id;
                    l.class_ids[truth_out_index] = class_id;
                    //printf(" track_id = %d, t = %d, b = %d, truth_in_index = %d, truth_out_index = %d \n", track_id, t, b, truth_in_index, truth_out_index);

                    // range is 0 <= 1
                    args->tot_iou += all_ious.iou;
                    args->tot_iou_loss += 1 - all_ious.iou;
//                    args->tot_iou_loss += all_ious.ciou;

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
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        ious all_ious = delta_yolo_box(update_delta, truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);

//                        acc_loss += all_ious.ciou;
                        acc_loss += 1 - all_ious.iou;
                        if(update_delta)
                        {
                            (*state.net.total_bbox)++;

                            // range is 0 <= 1
                            args->tot_iou += all_ious.iou;
                            args->tot_iou_loss += 1 - all_ious.iou;
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

                            }
                        }
                    }
                }

                //// COLOCAR Organizado por Loss

            }

//            if ((min_loss > acc_loss) && (acc_loss > 2*FLT_EPSILON)){
//                min_loss = acc_loss;
//                min_box_idx = t;
//            }

//            if(t_ant > t){
//                t = t_ant;
////                free(loss_group);
////                free(idx_group);
////                nelem_group = 1;
//            } else {
//                t++;
//            }

            if(all_evaluated==0){
                t++;
            }
//            if(final==1)
//                break;
            if((all_evaluated==1) && (current_group==num_boxes))
                break;
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
        }
        free(group_boxes);
        num_boxes = 0;

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

//        fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f), count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
//            (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, state.index, tot_iou / count, count, classification_loss, iou_loss, loss);

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
