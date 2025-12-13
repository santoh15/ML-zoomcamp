import pickle

model_file='model_c=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv,model=pickle.load(f_in)

#diccionario del cliente

