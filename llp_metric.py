import argparse
import json
import sed_eval
import dcase_util

def calc_metrics(predictions_json_file_path, gt_json_file_path):
    with open(predictions_json_file_path, 'r') as f:
        pred = json.load(f)
    with open(gt_json_file_path, 'r') as f:
        gt = json.load(f)
    
    reference_event_list = dcase_util.containers.MetaDataContainer(gt)
    estimated_event_list = dcase_util.containers.MetaDataContainer(pred)

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=1.0
    )
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=0.250
    )

    for filename in reference_event_list.unique_files:
        reference_event_list_for_current_file = reference_event_list.filter(
            filename=filename
        )

        estimated_event_list_for_current_file = estimated_event_list.filter(
            filename=filename
        )

        segment_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

        event_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    # Get only certain metrics
    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    print("Accuracy:", overall_segment_based_metrics['accuracy']['accuracy'])

    # Or print all metrics as reports
    print(segment_based_metrics)
    print(event_based_metrics)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_json_file_path', required=True, type=str)
    parser.add_argument('--gt_json_file_path', required=True, type=str)
    args = parser.parse_args()

    calc_metrics(args.predictions_json_file_path, args.gt_json_file_path)