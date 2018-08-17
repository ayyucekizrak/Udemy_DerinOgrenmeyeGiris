import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # ilk karakter büyük 
    print ('%s' % (txt, ), end='')

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def initialize_parameters(n_a, n_x, n_y):

    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # girişten saklıya
    Waa = np.random.randn(n_a, n_a)*0.01 # saklıdan saklıya
    Wya = np.random.randn(n_y, n_a)*0.01 # saklıdan çıkışa
    b = np.zeros((n_a, 1)) # saklı bias
    by = np.zeros((n_y, 1)) # çıkış bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # saklı durum
    p_t = softmax(np.dot(Wya, a_next) + by) # sonraki karakter için normalize edilmemiş log olasılıkları
    
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] 
    daraw = (1 - a * a) * da # doğrusal olmayan tanh fonksiyonunun gradyanı
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def rnn_forward(X, Y, a0, parameters, vocab_size = 27):
    
    # x, a ve y_hat boş sözlük olarak ilklendir
    x, a, y_hat = {}, {}, {}
    
    a[-1] = np.copy(a0)
    
    # yitimi sıfır olarak ilklendir
    loss = 0
    
    for t in range(len(X)):
        
        # x[t] one-hot vektör olacak şekilde ilklendir
        # X[t] == None olursa x[t]=0 olur. İlk girdi sıfır vektörü olmalıdır. 
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1
        
        # RNN'de bir adım ileri yönlü hesaplama gerçekleştir
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
        # Yitim değerini cross-entropy değerini çıkartarak güncelle
        loss -= np.log(y_hat[t][Y[t],0])
        
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    # Gradyanları boş sözlük olarak ilklendir
    gradients = {}
    
    # cache ve parametre değerlerini al
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    # gradyanlar aynı boyutta ve sıfır olarak ilklendirilmelidir
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    # Zamanda geriye yayılım
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    
    return gradients, a

