"""
network.py
~~~~~~~~~~
Un modulo per implementare l'algoritmo di learning a discesa di gradiente stocastica per una rete feedforward.
I gradienti sono calcolati tramite backpropagation.
Il codice non è ottimizzato, ma molto leggibile e facilmente modificabile.
"""

#### Librerie
# Librerie standard
import random

# Librerie di terze parti
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        La lista ''sizes'' contiene il numero di neuroni nei rispettivi
        livelli della rete. Per esempio:

        [               2,              3,              1               ]
                        |               |               |
                        |               |               |
                        V               V               V
                 2 neuroni lvl1   3 neuroni lvl2  1 neurone lvl3

        I bias e i pesi della rete sono inizializzati casualmente, usando
        una distribuzione gaussiana di media 0 e variabile 1. Notare il
        primo livello che si assume sia un livello di input, e che 
        convenzionalmente noi non setteremo bias per questi neuroni, siccome
        i bias sono usati nel computare gli output dai layer successivi.
        """
        self.num_layers = len(sizes) # Numero di layer = lunghezza(array sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # Prendi tutti gli elementi dal secondo in poi in ''sizes'' e mandali a ''randn()''
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] 
        """
        randn():
            1 - Prendi tutti gli elementi tranne il primo in ''sizes'' e mandali a ''randn()''
            2 - Prendi tutti gli elementi tranne l'ultimo in ''sizes'' e mandali a ''randn()''
        """

    def feedforward(self, a):
        """Ritorna l'output della rete se ``a`` è input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Allena la rete neurale usando la discesa stocastica a gradiente mini-batch.
        Il ''training_data'' è una lista di tuple ''(x, y)'' rappresentanti gli
        input del training e gli output desiderati. 
        Gli altri parametri non opzionali sono auto-esplicativi.
        Se ''test_data'' è stato dato allora la rete sarà verificata con i dati di
        test dopo ogni epoch e i progressi parziali andranno in output.
        Utile per vedere l'avanzamento, ma rallenta di molto il tutto.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        Aggiorna i pesi della rete ed i bias tramite l'applicazione del gradiente
        di discesa usando la backpropagation ad un singolo mini-batch.
        Il ''mini_batch'' è una lista di tuple ''(x, y)'', e ''eta'' è la rateo di learning.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Ritorna la tupla ''(nabla_b, nabla_w)'' rappresentanti il
        gradiente per la funzione di costo C_x. ''nabla_b'' e ''nabla_w''
        sono liste livello-per-livello di arrays numpy, simili a ''self.biases''
        e ''self.weigths''.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # Lista in cui memorizzare tutte le attivazioni, livello per livello
        zs = [] # Lista per memorizzare tutti i vettori z, livello per livello
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Da notare che la variabile l nel loop sotto è usata un po'
        # differentemente dalla notazione nel capitolo 2 del libro. Qua,
        # l = 1 = l'ultimo livello di neuroni, l = 2 = secondo lvl... e così
        # via. Il motivo di questa rinumerazione è il trarre vantaggio degli
        # indici negativi di Python nelle liste.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Ritorna il numero di test di input per i quali la rete
        esce il risultato corretto. Da notare che l'output della
        rete neurale si assume che sia l'indice di qualunque neurone
        nel livello finale con la più alta attivazione.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Ritorna il vettore di derivate parziali / parziali C_x / parziali a per l'attivazione dell'output.
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Funzioni miste
def sigmoid(z):
    """Funzione sigmoidale."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivata della funzione sigmoidale."""
    return sigmoid(z)*(1-sigmoid(z))
