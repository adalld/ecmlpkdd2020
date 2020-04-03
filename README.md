# Ada-LLD: Adaptive Node Similarity for Node Classification Using Multi-Scale Local Label Distribution

### This work is currently under review for ecml2020
This project contains the Python implementation for the work published in *Ada-LLD: Adaptive Node Similarity for Node Classification Using Multi-Scale Local Label Distribution*.
Please clone the project (alongside with the adopted ligra project from [here](https://github.com/adalld/ligra)) and set up your Python environment accordingly.

### Installation
To run the Ada-LLD project properly, you will need to install an adopted version of the Ligra project for the computation of the APPR vectors that are used to compute the input for the Ada-LLD models. Both, the Ada-LLD and the Ligra projects must be located in the same parent folder. The path to this parent folder must be provided with `base_path` argument when starter.py script is called. In fact, your folder structure should finally look like follows
```
path_to_parent/ligra
path_to_parent/adalld
```
with `path_to_parent` being the path to parent folder.  
Also we've used and tested in a Python 3.6 environment. The requirements can be found in `requirements.txt`.

Make sure g++ >= 5.3.0 compiler is installed

1. Clone the adopted version of the Ligra project from here
2. Go to apps folder `cd path_to_parent/ligra/apps` and run `make -j`
(Note: the -j option is used to parallelize the compiling process (see [here](https://github.com/jshun/ligra) for details),  
3. Switch to subfolder `cd path_to_parent/ligra/apps/localAlg` folder and run
`make -j` again.
4. Clone the Ada-LLD project into the same parent folder as the Ligra project (recall the details from above)
5. Go to project folder `cd path_to_parent/adadldd`.  Run `python src/starter.py --help` to verify that everything worked out.

For sample call and description of parameters see below


## Running Ada-LLD
The command line arguments for start script `src/starter.py` in Ada-LLD project are listed below:

| arg | description |
|---|---|
| &#x2011;&#x2011;dataset | The name of input dataset [Required]. This name is the same as folder name where dataset is stored in.|
| &#x2011;&#x2011;models | The models that shall run [Required]. We included `ld_avg` and `ld_concat`.`ld_indp` corresponds to `ld_concat` with concatenation in hidden layer and `ld_shared` is the same as `ld_indp` with shared weights, `2step_lp` (cf., [[1]](https://arxiv.org/pdf/1612.05001.pdf)). |
| &#x2011;&#x2011;base_path | The path pointing to the location that contains the `adalld` and the `ligra` project folders. The default is set properly for the image. |
| &#x2011;&#x2011;multilabel | This flag should be set when considering a multilabel task (the datasets for this are BlogCatalog and IMDb Germany). The default is set to false which means that multiclass tasks are considered per default. |
| &#x2011;&#x2011;plot_results | This flag should be set if the program shall create boxplots to visualize the results. png image files will be stored in the corresponding results folder. The default is false. |
| &#x2011;&#x2011;weighting_factor | This arg only takes effect for multilabel classification tasks. Weights the positive labels, as they are likely to be highly underrepresented. The default is set to 10.0 |
| &#x2011;&#x2011;instances_per_class | This arg takes an integer value as input and determines how many labeled instances of each class shall be considered for training. Default is set to 20. |
| &#x2011;&#x2011;training_fraction | This arg takes a float value (0.0, 1.0) as input and determines the fraction of data that shall be considered for training. Default is set to 0.7. *Note: instances_per_class and training_fraction are mutually exclusive.* |
| &#x2011;&#x2011;num_splits | Takes an integer value as input and determines the number of splits. Default is 10. |
| &#x2011;&#x2011;generate_splits | If this flag is used, new data splits will be generated. Otherwise, data splits are tried to be loaded from disk (which is the default). |
| &#x2011;&#x2011;runs | This arg expects an integer value determining the number of runs per split. Default is 5. |
| &#x2011;&#x2011;appr_alphas | A list of alphas values, i.e., floats in (0.0, 1.0), which determine the locality for the APPR computation.  |
| &#x2011;&#x2011;appr_epsilon | A float value (0.0, 1.0) which is the approximation threshold for the APPR computation. Default is 1E-4. |
| &#x2011;&#x2011;lp_alphas | A list of alphas values, i.e., floats in (0.0, 1.0), which determine the locality for the 2-step label propagation. |
| &#x2011;&#x2011;lp_betas | A list of beta values, i.e., integer values, which are used as step parameter for the 2-step label propagation. |

### Sample call
`python src/starter.py --dataset cora ‑‑generate_splits --models ld_avg ld_concat ld_indp ld_shared 2step_lp --plot_results --instances_per_class 20 --appr_alphas 0.1 0.5 0.9 --lp_alphas 0.1 0.5 0.9 --lp_betas 1 2 3`
  
This call runs all models on the Cora dataset with each model using 20 instances per class for training.
For the computation of the label distribution (i.e., the input for the Ada-LLD models), the alpha values {0.1, 0.5, 0.9} 
are used with the default approximation threshold 1E-4 for the computation of the APPR vectors (=*relevant neighborhoods*). 
For the 2-step label propagation approach, alpha values in {0.1, 0.5, 0.9} and beta values in {1, 2, 3} are used. 
The 2-step LP results plotted in the output image are the best results achieved by the grid search over the given settings. 
The results can be found at `/home/adalld/results/instances_per_class/20/`.  


### Incorporating own datasets
There is one folder for each dataset in `data` folder in Ada-LLD project.
To include own datasets into the project it is necessary to store the network and label data in new folder within `path_to_parent/adalld/data`. Graph has to provided in edge list format within a file named "edges.txt". 
The labels must be stored in scipy.sparse format in a file named "labels.npz". Also, you may need to register your dataset in the corresponding list of allowed datasets in the header
of the "starter.py" script.


## References
[1] Leto Peel. "Graph-based semi-supervised learning for relational networks." Proceedings of the 2017 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2017.  
[2] Thomas N. Kipf, and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).  
[3] Julian Shun and Guy E. Blelloch. "Ligra: A Lightweight Graph Processing Framework for Shared Memory." Proceedings of the ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP), pp. 135-146, 2013.

The documentation is under construction