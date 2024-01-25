The lime folder is a modiified version of the 0.2.0.1 version of [marcotcr/lime](https://github.com/marcotcr/lime) library.

The specific modification made is in the `explain_instance_with_data()` method of the `lime_base` module. In this method, the `easy_model` variable is made an instance variable of the class. This modification allows the evaluation code to access the `easy_model`.
