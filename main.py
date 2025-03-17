from functions import *

def main():
    main_train()    # Uncomment to train a model (need to implement 'sgd' function)
    main_predict('ner.dev', 'model')  # Uncomment to predict on 'dev.ner' using the model 'model' (need to implement 'decode' function)

if __name__ == "__main__":
    main()