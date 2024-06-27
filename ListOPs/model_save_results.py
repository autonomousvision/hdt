import json

def save_results(save_path, **kwargs):
    """
        Saves the results of the model, typically the test run, to a JSON file.
        Feel free to copy this into your own model code. 
        ExperimentRunner's Config.RESULTS_DIRECTORY and ../batch_jobs_completed/ must be the same directory. 
        This should work out of the box, if save_results is called from your project's directory. 
        Else you might need to change the path.

        :param save_name: The name of the file to save the results to. This is passed to the model by the ExperimentRunner.
        :param kwargs: The results to save. They should be named the same way as in your spredsheet e.g. test_accuracy=0.5
    """
    with open(save_path, 'w') as f:
        json.dump(kwargs, f)
    print(f'Results saved to {save_path}')

if __name__ == '__main__':
    save_results('test', test_accuracy=0.5) 