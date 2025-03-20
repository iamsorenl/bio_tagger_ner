from functions import main_train, main_predict, test_decoder

def main():
    main_train()    # Uncomment to train a model (need to implement 'sgd' function)
    main_predict('ner.dev', 'model')  # Uncomment to predict on 'dev.ner' using the model 'model' (need to implement 'decode' function)
    main_predict('ner.test', 'model')  # Uncomment to predict on 'test.ner' using the model 'model' (need to implement 'decode' function)
    
    #test_decoder()  # Uncomment to test the decoder (need to implement 'decode' function)

if __name__ == "__main__":
    main()