## Next steps

Now that you've finished the tutorial, you might be wondering: *what's next?*

### Collect your data

The first step to running your own search ranking experiments is to create a labelled training set. There are many techniques for doing this, ranging from having human annotators manually label your data, all the way to semi-automated labeling methods using machine learning.
You can also instrument your website to gather user clickthrough data and use that to create a training set.

### Deploy the model

1. The straightforward option is to save the model by following these steps in [XGBoost's Documentation](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html), host it alongside your Azure Cognitive Search client, and rerank queries locally.

2. For a hosted solution, you can check out [Azure Machine Learning](https://azure.microsoft.com/services/machine-learning/). Azure Machine Learning provides a variety of options for [training](https://docs.microsoft.com/azure/machine-learning/tutorial-1st-experiment-sdk-setup) and [hosting](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-existing-model) a model.

### Best practices for model deployment

- [Rules of Machine Learning](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) is a great technical checklist for model deployment.

- [Exp-Platform](https://exp-platform.com/) is a great resource from Microsoft's former head of Experimentation.
	- Gating features that can affect your search experience behind an A/B test, whether it's a UI change or a new search ranking model, will allow you to evaluate the model's performance in the real world.
	- For further reading, check out this whitepaper: [Online Controlled Experiments and A/B Tests](https://exp-platform.com/Documents/2015%20Online%20Controlled%20Experiments_EncyclopediaOfMLDM.pdf)
