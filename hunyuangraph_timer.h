#ifndef _H_TIME
#define _H_TIME

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_graph.h"

#include <cuda_runtime.h>

/*Time function params*/
// all time
double part_all = 0;
struct timeval begin_part_all;
struct timeval end_part_all;

// three phase
double part_coarsen = 0;
struct timeval begin_part_coarsen;
struct timeval end_part_coarsen;

double part_init = 0;
struct timeval begin_part_init;
struct timeval end_part_init;

double part_uncoarsen = 0;
struct timeval begin_part_uncoarsen;
struct timeval end_part_uncoarsen;

// steps
double part_match = 0;
struct timeval begin_part_match;
struct timeval end_part_match;

double part_contruction = 0;
struct timeval begin_part_contruction;
struct timeval end_part_contruction;

double part_cmatch = 0;
struct timeval begin_part_cmatch;
struct timeval end_part_cmatch;

double part_ccontract = 0;
struct timeval begin_part_ccontract;
struct timeval end_part_ccontract;

double part_bfs = 0;
struct timeval begin_part_bfs;
struct timeval end_part_bfs;

double part_2refine = 0;
struct timeval begin_part_2refine;
struct timeval end_part_2refine;

double part_2map = 0;
struct timeval begin_part_2map;
struct timeval end_part_2map;

double part_slipt = 0;
struct timeval begin_part_slipt;
struct timeval end_part_slipt;

double part_krefine = 0;
struct timeval begin_part_krefine;
struct timeval end_part_krefine;

double part_map = 0;
struct timeval begin_part_map;
struct timeval end_part_map;

double part_mallocrefine = 0;
struct timeval begin_part_mallocrefine;
struct timeval end_part_mallocrefine;

// test
double krefine_atomicadd = 0;
struct timeval begin_krefine_atomicadd;
struct timeval end_krefine_atomicadd;

// uncoarsen
double uncoarsen_Exnode_part1 = 0;
double uncoarsen_Exnode_part2 = 0;
struct timeval begin_general;
struct timeval   end_general;

double bndinfo_Find_real_bnd_info = 0;
double bndinfo_init_bnd_info = 0;
double bndinfo_find_kayparams = 0;
double bndinfo_initcucsr = 0;
double bndinfo_bb_segsort = 0;
double bndinfo_init_cu_que = 0;
double bndinfo_findcsr = 0;
struct timeval begin_bndinfo;
struct timeval end_bndinfo;

// match
double init_gpu_match_time = 0;
double check_length_time = 0;
double set_bin_time = 0;
double hem_gpu_match_time = 0;
double resolve_conflict_1_time = 0;
double resolve_conflict_2_time = 0;
double inclusive_scan_time1 = 0;
double resolve_conflict_4_time = 0;
struct timeval begin_gpu_match;
struct timeval end_gpu_match;

//match kernel
double match_time = 0;
double random_match_time = 0;
double init_gpu_receive_send_time = 0;
double wgt_segmentsort_gpu_time = 0;
double segmentsort_memcpy_time = 0;
double set_receive_send_time = 0;
double set_match_topk_time = 0;
double reset_match_array_time = 0;
double leaf_matches_step1_time = 0;
double leaf_matches_step2_time = 0;
double isolate_matches_time = 0;
double twin_matches_time = 0;
double relative_matches_step1_time = 0;
double relative_matches_step2_time = 0;
double match_malloc_time = 0;
double match_memcpy_time = 0;
double match_free_time = 0;
struct timeval begin_gpu_match_kernel;
struct timeval end_gpu_match_kernel;

//topk / four match time
double top1_time = 0;
double top2_time = 0;
double top3_time = 0;
double top4_time = 0;
double leaf_time = 0;
double isolate_time = 0;
double twin_time = 0;
double relative_time = 0;
struct timeval begin_gpu_topkfour_match;
struct timeval end_gpu_topkfour_match;

// contract
double exclusive_scan_time = 0;
double set_tadjncy_tadjwgt_time = 0;
double ncy_segmentsort_gpu_time = 0;
double mark_edges_time = 0;
double inclusive_scan_time2 = 0;
double set_cxadj_time = 0;
double init_cadjwgt_time = 0;
double set_cadjncy_cadjwgt_time = 0;
struct timeval begin_gpu_contraction;
struct timeval end_gpu_contraction;

double coarsen_malloc = 0;
struct timeval begin_malloc;
struct timeval end_malloc;

double coarsen_memcpy = 0;
struct timeval begin_coarsen_memcpy;
struct timeval end_coarsen_memcpy;

double coarsen_free = 0;
struct timeval begin_free;
struct timeval end_free;

double coarsen_else = 0;

// init
double set_initgraph_time = 0;
double initcurand_gpu_time = 0;
double bisection_gpu_time = 0;
double splitgraph_gpu_time = 0;
double select_where_gpu_time = 0;
double update_where_gpu_time = 0;
double update_answer_gpu_time = 0;
double update_tpwgts_time = 0;
double computecut_time = 0;
struct timeval begin_gpu_bisection;
struct timeval end_gpu_bisection;

double init_else = 0;

//  uncoarsen
double uncoarsen_initpwgts = 0;
double uncoarsen_calculateSum = 0;
double uncoarsen_projectback = 0;
double uncoarsen_Sum_maxmin_pwgts = 0;
double uncoarsen_select_init_select = 0;
double uncoarsen_select_bnd_vertices_warp = 0;
double uncoarsen_moving_interaction = 0;
double uncoarsen_update_select = 0;
double uncoarsen_execute_move = 0;
double uncoarsen_compute_edgecut = 0;
double uncoarsen_gpu_malloc = 0;
double uncoarsen_gpu_free = 0;
double uncoarsen_gpu_memcpy = 0;
struct timeval begin_gpu_kway;
struct timeval end_gpu_kway;
double uncoarsen_lp = 0;
double uncoarsen_rw = 0;
double uncoarsen_rs = 0;
double uncoarsen_pm = 0;
double uncoarsen_init_vals = 0;
double uncoarsen_set_gain_offset = 0;
double uncoarsen_prefixsum = 0;
double uncoarsen_set_gain_val = 0;
double uncoarsen_compute_imb = 0;
double uncoarsen_is_balance = 0;
double lp_init = 0;
double lp_select_dest_part = 0;
double lp_filter_potential_vertex = 0;
double lp_afterburner_heuristic = 0;
double lp_filter_beneficial_moves = 0;
double lp_set_lock = 0;
double pm_count_change1 = 0;
double pm_pm_cuda = 0;
double pm_update_large = 0;
double pm_count_change2 = 0;
struct timeval begin_gpu_kway_pm;
struct timeval end_gpu_kway_pm;
double pm_update_mark_adjacent = 0;
double pm_update_init = 0;
double pm_update_reset_conn_DS = 0;
struct timeval begin_gpu_kway_pm_update;
struct timeval end_gpu_kway_pm_update;
double rw_parts = 0;
double rw_select_dest_parts = 0;
double rw_init = 0;
double rw_assign_move_scores = 0;
double rw_prefixsum = 0;
double rw_filter_scores_below_cutoff = 0;
struct timeval begin_gpu_kway_rw;
struct timeval end_gpu_kway_rw;
double rs_init = 0;
double rs_assign_move_scores_part1 = 0;
double rs_prefixsum = 0;
double rs_assign_move_scores_part2 = 0;
double rs_assign_move_scores_part3 = 0;
double rs_find_score_cutoffs = 0;
double rs_filter_below_cutoffs = 0;
double rs_balance_scan_evicted_vertices = 0;
double rs_cookie_cutter = 0;
double rs_select_dest_parts = 0;
struct timeval begin_gpu_kway_rs;
struct timeval end_gpu_kway_rs;

double set_cpu_graph = 0;
struct timeval begin_set_cpu_graph;
struct timeval end_set_cpu_graph;

double set_graph_1 = 0;
double set_graph_2 = 0;
double set_graph_3 = 0;
double set_graph_4 = 0;
struct timeval begin_set_graph;
struct timeval end_set_graph;

double gpu_2way = 0;
struct timeval begin_gpu_2way;
struct timeval end_gpu_2way;

double malloc_2way = 0;
struct timeval begin_malloc_2way;
struct timeval end_malloc_2way;

double initmoveto = 0;
struct timeval begin_initmoveto;
struct timeval end_initmoveto;

double updatemoveto = 0;
struct timeval begin_updatemoveto;
struct timeval end_updatemoveto;

double computepwgts = 0;
struct timeval begin_computepwgts;
struct timeval end_computepwgts;

double thrustreduce = 0;
struct timeval begin_thrustreduce;
struct timeval end_thrustreduce;

double computegain = 0;
struct timeval begin_computegain;
struct timeval end_computegain;

double thrustsort = 0;
struct timeval begin_thrustsort;
struct timeval end_thrustsort;

double computegainv = 0;
struct timeval begin_computegainv;
struct timeval end_computegainv;

double inclusive = 0;
struct timeval begin_inclusive;
struct timeval end_inclusive;

double re_balance = 0;
struct timeval begin_rebalance;
struct timeval end_rebalance;

double malloc_split = 0;
struct timeval begin_malloc_split;
struct timeval end_malloc_split;

double memcpy_split = 0;
struct timeval begin_memcpy_split;
struct timeval end_memcpy_split;

double free_split = 0;
struct timeval begin_free_split;
struct timeval end_free_split;

double save_init = 0;
struct timeval begin_save_init;
struct timeval end_save_init;

void print_graph_infor(hunyuangraph_graph_t *graph, char *filename)
{
    printf("graph:%s %d %d\n", filename, graph->nvtxs, graph->nedges);
}

void init_timer()
{
    part_all = 0;
    part_coarsen = 0;
    part_init = 0;
    part_uncoarsen = 0;

    part_match = 0;
    part_contruction = 0;

    //  coarsen
    init_gpu_match_time = 0;
    check_length_time = 0;
    set_bin_time = 0;
    hem_gpu_match_time = 0;
    resolve_conflict_1_time = 0;
    resolve_conflict_2_time = 0;
    inclusive_scan_time1 = 0;
    resolve_conflict_4_time = 0;
    match_malloc_time = 0;
    match_memcpy_time = 0;
    match_free_time = 0;
    exclusive_scan_time = 0;
    set_tadjncy_tadjwgt_time = 0;
    ncy_segmentsort_gpu_time = 0;
    mark_edges_time = 0;
    inclusive_scan_time2 = 0;
    set_cxadj_time = 0;
    init_cadjwgt_time = 0;
    set_cadjncy_cadjwgt_time = 0;
    coarsen_malloc = 0;
    coarsen_memcpy = 0;
    coarsen_free = 0;

    // topk
    top1_time = 0;
    top2_time = 0;
    top3_time = 0;
    top4_time = 0;
    leaf_time = 0;
    isolate_time = 0;
    twin_time = 0;
    relative_time = 0;

    //  init
    set_initgraph_time = 0;
    initcurand_gpu_time = 0;
    bisection_gpu_time = 0;
    splitgraph_gpu_time = 0;
    select_where_gpu_time = 0;
    update_where_gpu_time = 0;
    update_answer_gpu_time = 0;
    update_tpwgts_time = 0;

    // uncoarsen
    uncoarsen_initpwgts = 0;
    uncoarsen_calculateSum = 0;
    uncoarsen_Sum_maxmin_pwgts = 0;
    uncoarsen_select_init_select = 0;
    uncoarsen_moving_interaction = 0;
    uncoarsen_update_select = 0;
    uncoarsen_execute_move = 0;
    uncoarsen_select_bnd_vertices_warp = 0;
    uncoarsen_projectback = 0;
    uncoarsen_gpu_malloc = 0;
    uncoarsen_gpu_free = 0;
    uncoarsen_gpu_memcpy = 0;
    uncoarsen_compute_edgecut = 0;

    uncoarsen_init_vals = 0;
    uncoarsen_compute_imb = 0;
    uncoarsen_set_gain_offset = 0;
    uncoarsen_prefixsum = 0;
    uncoarsen_set_gain_val = 0;    
    uncoarsen_is_balance = 0;  
    uncoarsen_lp = 0;
    uncoarsen_rw = 0;
    uncoarsen_rs = 0;
    uncoarsen_pm = 0;

    lp_init = 0;
    lp_select_dest_part = 0;
    lp_filter_potential_vertex = 0;
    lp_afterburner_heuristic = 0;
    lp_filter_beneficial_moves = 0;
    lp_set_lock = 0;

    pm_count_change1 = 0;
    pm_pm_cuda = 0;
    pm_update_large = 0;
    pm_count_change2 = 0;
    pm_update_mark_adjacent = 0;
    pm_update_init = 0;
    pm_update_reset_conn_DS = 0;

    rw_parts = 0;
    rw_select_dest_parts = 0;
    rw_init = 0;
    rw_assign_move_scores = 0;
    rw_prefixsum = 0;
    rw_filter_scores_below_cutoff = 0;

    rs_init = 0;
    rs_assign_move_scores_part1 = 0;
    rs_prefixsum = 0;
    rs_assign_move_scores_part2 = 0;
    rs_assign_move_scores_part3 = 0;
    rs_find_score_cutoffs = 0;
    rs_filter_below_cutoffs = 0;
    rs_balance_scan_evicted_vertices = 0;
    rs_cookie_cutter = 0;
    rs_select_dest_parts = 0;
}

void print_time_all(hunyuangraph_graph_t *graph, int *part, int edgecut, float imbalance)
{
    printf("---------------------------------------------------------\n");
    printf("Hunyuangraph-Partition-end\n");
    printf("Hunyuangraph_Partition_time= %10.2lf ms\n", part_all);
    printf("------Coarsen_time=          %10.2lf ms\n", part_coarsen);
    printf("------Init_time=             %10.2lf ms\n", part_init);
    printf("------Uncoarsen_time=        %10.2lf ms\n", part_uncoarsen);
    printf("------else_time=             %10.2lf ms\n", part_all - (part_coarsen + part_init + part_uncoarsen));
    printf("edge-cut=                    %10d\n", edgecut);
    printf("imbalance=                   %10.3f\n", imbalance);
    printf("---------------------------------------------------------\n");
}

void print_time_coarsen()
{
    printf("\n");

    coarsen_else = part_coarsen - (init_gpu_match_time + check_length_time + set_bin_time + hem_gpu_match_time + resolve_conflict_1_time + resolve_conflict_2_time + inclusive_scan_time1 +
                                        resolve_conflict_4_time - match_malloc_time - match_memcpy_time - match_free_time + \
                                   exclusive_scan_time + set_tadjncy_tadjwgt_time + ncy_segmentsort_gpu_time + mark_edges_time + inclusive_scan_time2 +
                                    set_cxadj_time + init_cadjwgt_time + set_cadjncy_cadjwgt_time + coarsen_malloc + coarsen_memcpy + coarsen_free);
    printf("---------------------------------------------------------\n");
    printf("Coarsen_time=              %10.3lf ms\n", part_coarsen);
    printf("    part_match                 %10.3lf %7.3lf%\n", part_match, part_match / part_coarsen * 100);
    printf("        init_gpu_match_time        %10.3lf %7.3lf%\n", init_gpu_match_time, init_gpu_match_time / part_coarsen * 100);
    printf("        check_length_time          %10.3lf %7.3lf%\n", check_length_time, check_length_time / part_coarsen * 100);
    printf("        set_bin_time               %10.3lf %7.3lf%\n", set_bin_time, set_bin_time / part_coarsen * 100);
    printf("        hem_gpu_match_time         %10.3lf %7.3lf%\n", hem_gpu_match_time, hem_gpu_match_time / part_coarsen * 100);
    printf("            random_match_time          %10.3lf %7.3lf%\n", random_match_time, random_match_time / hem_gpu_match_time * 100);
    printf("            init_gpu_receive_send_time %10.3lf %7.3lf%\n", init_gpu_receive_send_time, init_gpu_receive_send_time / hem_gpu_match_time * 100);
    printf("            wgt_segmentsort_gpu_time   %10.3lf %7.3lf%\n", wgt_segmentsort_gpu_time, wgt_segmentsort_gpu_time / hem_gpu_match_time * 100);
    printf("            segmentsort_memcpy_time    %10.3lf %7.3lf%\n", segmentsort_memcpy_time, segmentsort_memcpy_time / hem_gpu_match_time * 100);
    printf("            set_receive_send_time      %10.3lf %7.3lf%\n", set_receive_send_time, set_receive_send_time / hem_gpu_match_time * 100);
    printf("            set_match_topk_time        %10.3lf %7.3lf%\n", set_match_topk_time, set_match_topk_time / hem_gpu_match_time * 100);
    printf("            reset_match_array_time     %10.3lf %7.3lf%\n", reset_match_array_time, reset_match_array_time / hem_gpu_match_time * 100);
    printf("            leaf_matches               %10.3lf %7.3lf%\n", leaf_matches_step1_time + leaf_matches_step2_time, (leaf_matches_step1_time + leaf_matches_step2_time) / hem_gpu_match_time * 100);
    printf("                step1                      %10.3lf %7.3lf%\n", leaf_matches_step1_time, leaf_matches_step1_time / hem_gpu_match_time * 100);
    printf("                step2                      %10.3lf %7.3lf%\n", leaf_matches_step2_time, leaf_matches_step2_time / hem_gpu_match_time * 100);
    printf("            isolate_matches            %10.3lf %7.3lf%\n", isolate_matches_time, isolate_matches_time / hem_gpu_match_time * 100);
    printf("            twin_matches               %10.3lf %7.3lf%\n", twin_matches_time, twin_matches_time / hem_gpu_match_time * 100);
    printf("            relative_matches           %10.3lf %7.3lf%\n", relative_matches_step1_time + relative_matches_step2_time, (relative_matches_step1_time + relative_matches_step2_time) / hem_gpu_match_time * 100);
    printf("                step1                      %10.3lf %7.3lf%\n", relative_matches_step1_time, relative_matches_step1_time / hem_gpu_match_time * 100);
    printf("                step2                      %10.3lf %7.3lf%\n", relative_matches_step2_time, relative_matches_step2_time / hem_gpu_match_time * 100);
    printf("            match_malloc_time          %10.3lf %7.3lf%\n", match_malloc_time, match_malloc_time / hem_gpu_match_time * 100);
    printf("            match_memcpy_time          %10.3lf %7.3lf%\n", match_memcpy_time, match_memcpy_time / hem_gpu_match_time * 100);
    printf("            match_free_time            %10.3lf %7.3lf%\n", match_free_time, match_free_time / hem_gpu_match_time * 100);
    printf("        resolve_conflict_1_time    %10.3lf %7.3lf%\n", resolve_conflict_1_time, resolve_conflict_1_time / part_coarsen * 100);
    printf("        resolve_conflict_2_time    %10.3lf %7.3lf%\n", resolve_conflict_2_time, resolve_conflict_2_time / part_coarsen * 100);
    printf("        inclusive_scan_time        %10.3lf %7.3lf%\n", inclusive_scan_time1, inclusive_scan_time1 / part_coarsen * 100);
    printf("        resolve_conflict_4_time    %10.3lf %7.3lf%\n", resolve_conflict_4_time, resolve_conflict_4_time / part_coarsen * 100);
    printf("    part_contruction           %10.3lf %7.3lf%\n", part_contruction, part_contruction / part_coarsen * 100);
    printf("        exclusive_scan_time        %10.3lf %7.3lf%\n", exclusive_scan_time, exclusive_scan_time / part_coarsen * 100);
    printf("        set_tadjncy_tadjwgt_time   %10.3lf %7.3lf%\n", set_tadjncy_tadjwgt_time, set_tadjncy_tadjwgt_time / part_coarsen * 100);
    printf("        ncy_segmentsort_gpu_time   %10.3lf %7.3lf%\n", ncy_segmentsort_gpu_time, ncy_segmentsort_gpu_time / part_coarsen * 100);
    printf("        mark_edges_time            %10.3lf %7.3lf%\n", mark_edges_time, mark_edges_time / part_coarsen * 100);
    printf("        inclusive_scan_time2       %10.3lf %7.3lf%\n", inclusive_scan_time2, inclusive_scan_time2 / part_coarsen * 100);
    printf("        set_cxadj_time             %10.3lf %7.3lf%\n", set_cxadj_time, set_cxadj_time / part_coarsen * 100);
    printf("        init_cadjwgt_time          %10.3lf %7.3lf%\n", init_cadjwgt_time, init_cadjwgt_time / part_coarsen * 100);
    printf("        set_cadjncy_cadjwgt_time   %10.3lf %7.3lf%\n", set_cadjncy_cadjwgt_time, set_cadjncy_cadjwgt_time / part_coarsen * 100);
    printf("    coarsen_malloc             %10.3lf %7.3lf%\n", coarsen_malloc, coarsen_malloc / part_coarsen * 100);
    printf("    coarsen_memcpy             %10.3lf %7.3lf%\n", coarsen_memcpy, coarsen_memcpy / part_coarsen * 100);
    printf("    coarsen_free               %10.3lf %7.3lf%\n", coarsen_free, coarsen_free / part_coarsen * 100);
    printf("    else                       %10.3lf %7.3lf%\n", coarsen_else, coarsen_else / part_coarsen * 100);
    printf("---------------------------------------------------------\n");
}

void print_time_topkfour_match()
{
    double all = top1_time + top2_time + top3_time + top4_time + leaf_time + isolate_time + twin_time + relative_time;
    
    printf("---------------------------------------------------------\n");
    printf("all                        %10.3lf ms\n", all);
    printf("top1_time                  %10.3lf %7.3lf%\n", top1_time, top1_time / all * 100);
    printf("top2_time                  %10.3lf %7.3lf%\n", top2_time, top2_time / all * 100);
    printf("top3_time                  %10.3lf %7.3lf%\n", top3_time, top3_time / all * 100);
    printf("top4_time                  %10.3lf %7.3lf%\n", top4_time, top4_time / all * 100);
    printf("leaf_time                  %10.3lf %7.3lf%\n", leaf_time, leaf_time / all * 100);
    printf("isolate_time               %10.3lf %7.3lf%\n", isolate_time, isolate_time / all * 100);
    printf("twin_time                  %10.3lf %7.3lf%\n", twin_time, twin_time / all * 100);
    printf("relative_time              %10.3lf %7.3lf%\n", relative_time, relative_time / all * 100);
    printf("---------------------------------------------------------\n");
}

void print_time_init()
{
    printf("\n");

    init_else = part_init - (set_initgraph_time + initcurand_gpu_time + bisection_gpu_time + splitgraph_gpu_time + select_where_gpu_time + update_where_gpu_time + update_answer_gpu_time + 
                             update_tpwgts_time);
    
    printf("---------------------------------------------------------\n");
    printf("Init_time=                 %10.3lf ms\n", part_init);
    printf("    set_initgraph_time         %10.3lf %7.3lf%\n", set_initgraph_time, set_initgraph_time / part_init * 100);
    printf("    update_tpwgts_time         %10.3lf %7.3lf%\n", update_tpwgts_time, update_tpwgts_time / part_init * 100);
    printf("    initcurand_gpu_time        %10.3lf %7.3lf%\n", initcurand_gpu_time, initcurand_gpu_time / part_init * 100);
    printf("    gpu_Bisection_time         %10.3lf %7.3lf%\n", bisection_gpu_time, bisection_gpu_time / part_init * 100);
    printf("    splitgraph_gpu_time        %10.3lf %7.3lf%\n", splitgraph_gpu_time, splitgraph_gpu_time / part_init * 100);
    printf("    select_where_gpu_time      %10.3lf %7.3lf%\n", select_where_gpu_time, select_where_gpu_time / part_init * 100);
    printf("    update_where_gpu_time      %10.3lf %7.3lf%\n", update_where_gpu_time, update_where_gpu_time / part_init * 100);
    printf("    update_answer_gpu_time     %10.3lf %7.3lf%\n", update_answer_gpu_time, update_answer_gpu_time / part_init * 100);
    printf("    else                       %10.3lf %7.3lf%\n", init_else, init_else / part_init * 100);

    // printf("    computecut_time            %10.3lf %7.3lf%\n", computecut_time, computecut_time / part_init * 100);
    printf("---------------------------------------------------------\n");
}

void print_time_uncoarsen()
{
    printf("\n");
    double Uncoarsen_else = part_uncoarsen - (uncoarsen_init_vals + uncoarsen_compute_imb + uncoarsen_set_gain_offset + uncoarsen_prefixsum + uncoarsen_set_gain_val + \
                                              uncoarsen_is_balance + uncoarsen_lp + uncoarsen_rw + uncoarsen_rs + uncoarsen_pm + \
                                              uncoarsen_projectback + uncoarsen_gpu_malloc + uncoarsen_gpu_free + uncoarsen_gpu_memcpy + uncoarsen_compute_edgecut);

    printf("---------------------------------------------------------\n");
    printf("Uncoarsen_time=            %10.3lf ms\n", part_uncoarsen);
    printf("Uncoarsen init_vals         %10.3lf %7.3lf%\n", uncoarsen_init_vals, uncoarsen_init_vals / part_uncoarsen * 100);
    printf("Uncoarsen compute_imb          %10.3lf %7.3lf%\n", uncoarsen_compute_imb, uncoarsen_compute_imb / part_uncoarsen * 100);
    printf("Uncoarsen set_gain_offset      %10.3lf %7.3lf%\n", uncoarsen_set_gain_offset, uncoarsen_set_gain_offset / part_uncoarsen * 100);
    printf("Uncoarsen prefixsum            %10.3lf %7.3lf%\n", uncoarsen_prefixsum, uncoarsen_prefixsum / part_uncoarsen * 100);
    printf("Uncoarsen set_gain_val         %10.3lf %7.3lf%\n", uncoarsen_set_gain_val, uncoarsen_set_gain_val / part_uncoarsen * 100);
    printf("Uncoarsen is_balance           %10.3lf %7.3lf%\n", uncoarsen_is_balance, uncoarsen_is_balance / part_uncoarsen * 100);
    printf("Uncoarsen lp                   %10.3lf %7.3lf%\n", uncoarsen_lp, uncoarsen_lp / part_uncoarsen * 100);
    printf("Uncoarsen lp_init                  %10.3lf %7.3lf%\n", lp_init, lp_init / uncoarsen_lp * 100);
    printf("Uncoarsen lp_select_dest_part      %10.3lf %7.3lf%\n", lp_select_dest_part, lp_select_dest_part / uncoarsen_lp * 100);
    printf("Uncoarsen lp_filter_potential      %10.3lf %7.3lf%\n", lp_filter_potential_vertex, lp_filter_potential_vertex / uncoarsen_lp * 100);
    printf("Uncoarsen lp_afterburner_heuristic %10.3lf %7.3lf%\n", lp_afterburner_heuristic, lp_afterburner_heuristic / uncoarsen_lp * 100);
    printf("Uncoarsen lp_filter_beneficial_    %10.3lf %7.3lf%\n", lp_filter_beneficial_moves, lp_filter_beneficial_moves / uncoarsen_lp * 100);
    printf("Uncoarsen lp_set_lock              %10.3lf %7.3lf%\n", lp_set_lock, lp_set_lock / uncoarsen_lp * 100);
    printf("Uncoarsen rw                   %10.3lf %7.3lf%\n", uncoarsen_rw, uncoarsen_rw / part_uncoarsen * 100);
    printf("Uncoarsen rw_init                  %10.3lf %7.3lf%\n", rw_init, rw_init / uncoarsen_rw * 100);
    printf("Uncoarsen rw_parts                 %10.3lf %7.3lf%\n", rw_parts, rw_parts / uncoarsen_rw * 100);
    printf("Uncoarsen rw_select_dest_parts     %10.3lf %7.3lf%\n", rw_select_dest_parts, rw_select_dest_parts / uncoarsen_rw * 100);
    printf("Uncoarsen rw_assign_move_scores    %10.3lf %7.3lf%\n", rw_assign_move_scores, rw_assign_move_scores / uncoarsen_rw * 100);
    printf("Uncoarsen rw_prefixsum             %10.3lf %7.3lf%\n", rw_prefixsum, rw_prefixsum / uncoarsen_rw * 100);
    printf("Uncoarsen rw_filter_scores_        %10.3lf %7.3lf%\n", rw_filter_scores_below_cutoff, rw_filter_scores_below_cutoff / uncoarsen_rw * 100);
    printf("Uncoarsen rs                   %10.3lf %7.3lf%\n", uncoarsen_rs, uncoarsen_rs / part_uncoarsen * 100);
    printf("Uncoarsen rs_init                  %10.3lf %7.3lf%\n", rs_init, rs_init / uncoarsen_rs * 100);
    printf("Uncoarsen rs_assign_move_1         %10.3lf %7.3lf%\n", rs_assign_move_scores_part1, rs_assign_move_scores_part1 / uncoarsen_rs * 100);
    printf("Uncoarsen rs_prefixsum             %10.3lf %7.3lf%\n", rs_prefixsum, rs_prefixsum / uncoarsen_rs * 100);
    printf("Uncoarsen rs_assign_move_2         %10.3lf %7.3lf%\n", rs_assign_move_scores_part2, rs_assign_move_scores_part2 / uncoarsen_rs * 100);
    printf("Uncoarsen rs_assign_move_3         %10.3lf %7.3lf%\n", rs_assign_move_scores_part3, rs_assign_move_scores_part3 / uncoarsen_rs * 100);
    printf("Uncoarsen rs_find_score_cutoffs    %10.3lf %7.3lf%\n", rs_find_score_cutoffs, rs_find_score_cutoffs / uncoarsen_rs * 100);
    printf("Uncoarsen rs_filter_below_cutoffs  %10.3lf %7.3lf%\n", rs_filter_below_cutoffs, rs_filter_below_cutoffs / uncoarsen_rs * 100);
    printf("Uncoarsen rs_balance_scan_         %10.3lf %7.3lf%\n", rs_balance_scan_evicted_vertices, rs_balance_scan_evicted_vertices / uncoarsen_rs * 100);
    printf("Uncoarsen rs_cookie_cutter         %10.3lf %7.3lf%\n", rs_cookie_cutter, rs_cookie_cutter / uncoarsen_rs * 100);
    printf("Uncoarsen rs_select_dest_parts     %10.3lf %7.3lf%\n", rs_select_dest_parts, rs_select_dest_parts / uncoarsen_rs * 100);
    printf("Uncoarsen pm                   %10.3lf %7.3lf%\n", uncoarsen_pm, uncoarsen_pm / part_uncoarsen * 100);
    printf("Uncoarsen pm_count_change1         %10.3lf %7.3lf%\n", pm_count_change1, pm_count_change1 / uncoarsen_pm * 100);
    printf("Uncoarsen pm_pm_cuda               %10.3lf %7.3lf%\n", pm_pm_cuda, pm_pm_cuda / uncoarsen_pm * 100);
    printf("Uncoarsen pm_update_large          %10.3lf %7.3lf%\n", pm_update_large, pm_update_large / uncoarsen_pm * 100);
    printf("Uncoarsen pm_init                      %10.3lf %7.3lf%\n", pm_update_init, pm_update_init / pm_update_large * 100);
    printf("Uncoarsen pm_mark_adjacent             %10.3lf %7.3lf%\n", pm_update_mark_adjacent, pm_update_mark_adjacent / pm_update_large * 100);
    printf("Uncoarsen pm_reset_conn_DS             %10.3lf %7.3lf%\n", pm_update_reset_conn_DS, pm_update_reset_conn_DS / pm_update_large * 100);
    printf("Uncoarsen pm_count_change2         %10.3lf %7.3lf%\n", pm_count_change2, pm_count_change2 / uncoarsen_pm * 100);
    printf("Uncoarsen projectback          %10.3lf %7.3lf%\n", uncoarsen_projectback, uncoarsen_projectback / part_uncoarsen * 100);
    printf("Uncoarsen malloc               %10.3lf %7.3lf%\n", uncoarsen_gpu_malloc, uncoarsen_gpu_malloc / part_uncoarsen * 100);
    printf("Uncoarsen free                 %10.3lf %7.3lf%\n", uncoarsen_gpu_free, uncoarsen_gpu_free / part_uncoarsen * 100);
    printf("Uncoarsen memcpy               %10.3lf %7.3lf%\n", uncoarsen_gpu_memcpy, uncoarsen_gpu_memcpy / part_uncoarsen * 100);
    printf("Uncoarsen compute_edgecut      %10.3lf %7.3lf%\n", uncoarsen_compute_edgecut, uncoarsen_compute_edgecut / part_uncoarsen * 100);
    printf("Uncoarsen else                 %10.3lf %7.3lf%\n", Uncoarsen_else, Uncoarsen_else / part_uncoarsen * 100);
    printf("---------------------------------------------------------\n");

    // printf("Uncoarsen initpwgts            %10.3lf %7.3lf%\n", uncoarsen_initpwgts, uncoarsen_initpwgts / part_uncoarsen * 100);
    // printf("Uncoarsen calculateSum         %10.3lf %7.3lf%\n", uncoarsen_calculateSum, uncoarsen_calculateSum / part_uncoarsen * 100);
    // printf("Uncoarsen Sum_maxmin_pwgts     %10.3lf %7.3lf%\n", uncoarsen_Sum_maxmin_pwgts, uncoarsen_Sum_maxmin_pwgts / part_uncoarsen * 100);
    // printf("Uncoarsen select_init          %10.3lf %7.3lf%\n", uncoarsen_select_init_select, uncoarsen_select_init_select / part_uncoarsen * 100);
    // printf("Uncoarsen select_bnd           %10.3lf %7.3lf%\n", uncoarsen_select_bnd_vertices_warp, uncoarsen_select_bnd_vertices_warp / part_uncoarsen * 100);
    // printf("Uncoarsen moving_interaction   %10.3lf %7.3lf%\n", uncoarsen_moving_interaction, uncoarsen_moving_interaction / part_uncoarsen * 100);
    // printf("Uncoarsen update_select        %10.3lf %7.3lf%\n", uncoarsen_update_select, uncoarsen_update_select / part_uncoarsen * 100);
    // printf("Uncoarsen execute_move         %10.3lf %7.3lf%\n", uncoarsen_execute_move, uncoarsen_execute_move / part_uncoarsen * 100);
}

#endif