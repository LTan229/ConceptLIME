# ConceptLIME: Concept-based Local Interpretable Model-agnostic Explanations

ConceptLIME (Concept-based Local Interpretable Model-agnostic Explanations) is an explainable artificial intelligence (XAI) method that provides local explanations by constructing surrogate models in the vicinity of the explained sample. Unlike the previous LIME (Local Interpretable Model-agnostic Explanations) method, which works on the super-pixel level, ConceptLIME leverages concept-level perturbations to formulate explanations at the concept level, enhancing their comprehensibility to humans.

This repository hosts an implementation of ConceptLIME and the code used to generate the experimental results presented in the accompanying paper. 

We plan to perform more comprehensive experiments by comparing our method with additional state-of-the-art methods and evaluating them on a broader range of datasets.

Please refer to the paper for detailed information on ConceptLIME.

Contributions, bug reports, and feature requests are welcome.

## Usage

The project is developed using Python 3.8 and relies on specific versions of dependencies as specified in the `requirements.txt` file. 

Before running the code, please download the necessary models and datasets following the steps provided in the `prerequisites.ipynb` notebook.



