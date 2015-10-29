import numpy as np
import theano
import theano.tensor as T
from theano import function, printing, pp
import sys, random, pprint
from theano.compile.nanguardmode import NanGuardMode

from theano_util import *
from keras.activations import tanh, hard_sigmoid
from keras.initializations import glorot_uniform, orthogonal
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
from keras.preprocessing import sequence

from qa_dataset_parser import parse_qa_dataset

import cPickle

import skipthoughts


# theano.config.exception_verbosity = 'high'
# theano.config.allow_gc = False
#theano.config.profile = True
'''
wmemNN = WMemNN(n_words=num_words, n_embedding=4800, lr=0.01, word_to_id=word_to_id, null_word_id=null_word_id,
                        max_stmts=max_stmts, max_words=max_words)
'''
def inspect_inputs(i, node, fn):
    print i, node, "inputs:", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print i, node, "outputs:", [output[0] for output in fn.outputs]


# This is the current suggested detect_nan implementation to
# show you how it work.  That way, you can modify it for your
# need.  If you want exactly this method, you can use
# ``theano.compile.monitormode.detect_nan`` that will always
# contain the current suggested version.

'''
def detect_nan(i, node, fn):
    for input in fn.inputs:
            print '*** NaN detected ***'
            #theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break
'''


def detect_nan(i, node, fn):
    for input in fn.inputs:
        if (not isinstance(input[0], np.random.RandomState) and
            np.isnan(input[0]).any()):
            #input[0] = np.nan_to_num(input[0])
            #input = np.nan_to_num(input)
            print '*** NaN detected ***'
            #theano.printing.debugprint(node)
            #theano.printing.pp(fn.inputs)
            
            
            print "start train_function_print"
            theano.printing.pydotprint(wmemNN.train_function, outfile="/home/loop/DeepLearningLibraries/memnn_skip_thoughts/pics/symbolic_graph_unopt.png", var_with_name_simple=True)
            print "end train_function_print"
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print "start train_function_print"
            theano.printing.pydotprint(cost, outfile="/home/loop/DeepLearningLibraries/memnn_skip_thoughts/pics/symbolic_graph_unopt.png", var_with_name_simple=True)
            print "end train_function_print"
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            print "start train_function_print"
            theano.printing.pydotprint(cost, outfile="/home/loop/DeepLearningLibraries/memnn_skip_thoughts/pics/symbolic_graph_unopt.png", var_with_name_simple=True)
            print "end train_function_print"
            

            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            theano.printing.pp(fn.inputs)
            break


'''
x = theano.tensor.dscalar('x')
f = theano.function([x], [theano.tensor.log(x) * x],
                    mode=theano.compile.MonitorMode(
                        post_func=detect_nan))
f(0)  # log(0) * 0 = -inf * 0 = NaN
'''
'''
class GradClip(theano.compile.ViewOp):

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]

grad_clip = GradClip(-1.0, 200.0)
register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip')
'''

class WMemNN:
    '''
    def __init__(self, n_words=110, n_embedding=550, lr=0.01,
                 momentum=0.9, word_to_id=None, null_word_id=-1,
                 max_stmts=110, max_words=4800, load_from_file=None):
    '''
    #def __init__(self, n_words=110, n_embedding=100, lr=0.01,
    def __init__(self, n_words=110, n_embedding=4800, lr=0.01,
                 momentum=0.9, word_to_id=None, null_word_id=-1,
                 max_stmts=110, max_words=4800, load_from_file=None):
        if load_from_file:
            self.load_model(load_from_file)
        else:
            #self.regularization = 0.001
            self.regularization = 0.000001
            self.n_embedding = n_embedding
            self.lr = lr
            self.momentum = momentum
            self.n_words = n_words
            self.batch_size = 4
            self.max_stmts = max_stmts
            self.max_words = max_words

            self.word_to_id = word_to_id
            self.id_to_word = dict((v, k) for k, v in word_to_id.iteritems())
            self.null_word_id = null_word_id

            # Question embedding
            # self.B = init_shared_normal(self.n_words, self.n_embedding, 0.1)

            # Statement input, output embeddings
            #self.weights = init_shared_normal_tensor(4, self.n_words, self.n_embedding, 0.1)
            #self.weights = init_shared_normal_tensor(4, 4800, 100, 0.1)
            #self.weights = init_shared_normal_tensor(110, 4800, 1, 0.1)
            #self.weights = init_shared_normal_tensor(110, 4800, 80, 0.1)
            self.weights = init_shared_normal_tensor(110, 80, 4800, 0.1)
            #self.weights = init_shared_normal_tensor(110, 110, 4800, 0.1)
            #self.weights = init_shared_normal_tensor(110, 4800, 4800, 0.1)
            #You're not sure what self.n_embedding (80 in this case) does

            # Linear mapping between layers
            #self.H = init_shared_normal(self.n_embedding, 80, 0.1)
            self.H = init_shared_normal(self.n_embedding, self.n_embedding, 0.1)

            # Final outut weight matrix
            # self.W = init_shared_normal(self.n_embedding, self.n_words, 0.1)


        zero_vector = T.vector('zv', dtype=theano.config.floatX)

        # Statement
        #x = T.dvector('x')
        #x = T.dmatrix('x')
        x = T.dtensor3('x') #1 w3
        #x = T.dtensor4('x') #3 w3
        xbatch = T.dtensor4('xb')
        #xbatch = T.dtensor3('xb')
        #xbatch = T.dmatrix('xb')
        #xbatch = T.dvector('xb')
        #self.X = T.dscalar('X')
        #self.X = T.dtensor3('X') #3 w3 for both
        #xbatch = T.matrix('xb', dtype='float64')
        #xbatch = T.tensor3('xb', dtype='int32')
        #xbatch = T.tensor3('xb')

        # Positional encoding matrix
        pe = T.tensor3('pe')
        
        #Skip-thought encoding matrix if you want to add that
        #model = skipthoughts.load_model()
        #vectors = skipthoughts.encode(model, X)

        # Question
        q = T.dvector('q')
        #q = T.fvector('q')
        qbatch = T.dmatrix('qb')
        #qbatch = T.fmatrix('qb')

        # True word
        r = T.iscalar('r')
        rbatch = T.ivector('rb')

        
        memory_cost = self.memnn_cost(x, q, pe)
        print "mc start"
        print memory_cost
        print "mc end"
        
        print "weight_shape"
        print self.weights.shape[0]

        # memory_loss = -T.log(memory_cost[r]) # cross entropy on softmax
        memory_loss = self.memnn_batch_cost(xbatch, qbatch, rbatch, pe)

        params = [
            self.weights,
            # self.B,
            # self.W,
            self.H
        ]

        
        regularization_cost = reduce(
            lambda x,y: x + y,
            map(lambda x: self.regularization * T.sum(x ** 2), params)
        )
        
        cost = memory_loss + regularization_cost
        #cost = memory_loss
        #cost = regularization_cost
        

    

        
        print "regularization_cost"
        print regularization_cost

        #theano.printing.Print("this is reg cost")(regularization_cost)
        #print g
        
        
        print "memory_loss"
        print memory_loss

        print "cost_way_in"
        print cost
        

        theano.printing.Print("this is memory loss")(memory_loss)
        #print k

        #params = theano.gradient.grad_clip(params, -1, 200)

        grads = T.grad(cost, params)
        

        #grads = T.grad(cost, theano.gradient.grad_clip(params, -1, 200))
        #updates = [(param, param - learning_rate * grad_clip(-1, 1, gparam)) for param, gparam in zip(net.params, grad_params)]

        #params = theano.gradient.grad_clip(params, -1, 200)

        l_rate = T.scalar('l_rate')

        # Parameter updates
        updates = get_param_updates(params, grads, lr=l_rate, method='adadelta', momentum=0.9,
            constraint=self._constrain_embedding(self.null_word_id, zero_vector))

        self.train_function = theano.function(
            inputs = [
                #xbatch[0], qbatch, rbatch, pe,
                xbatch, qbatch, rbatch, pe,
                theano.Param(l_rate, default=self.lr),
                theano.Param(zero_vector, default=np.zeros((self.n_embedding,), theano.config.floatX))
            ],
            outputs = cost,
            updates = updates,
            allow_input_downcast=True,
            # mode='FAST_COMPILE',
            #mode='DebugMode'
            #mode=theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs)
            #mode=theano.compile.MonitorMode(pre_func=detect_nan, post_func=detect_nan)
            #mode=theano.compile.MonitorMode(pre_func=detect_nan, post_func=detect_nan).excluding('local_elemwise_fusion', 'inplace')
            #mode=theano.compile.MonitorMode(pre_func=detect_nan, post_func=detect_nan).excluding('local_elemwise_fusion')
            #mode=theano.compile.MonitorMode(pre_func=detect_nan, post_func=detect_nan).excluding('inplace')
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
            on_unused_input='warn'
        )

        '''
        x = theano.tensor.dscalar('x')
        f = theano.function([x], [5 * x],
                            mode=theano.compile.MonitorMode(
                                pre_func=inspect_inputs,
                                post_func=inspect_outputs))
        f(3)
        '''

        self.predict_function = theano.function(
            inputs = [
                x, q, pe
                #[x], q, pe
            ],
            outputs = memory_cost,
            allow_input_downcast=True,
            # mode='FAST_COMPILE',
            on_unused_input='warn'
        )

    def _constrain_embedding(self, null_id, zero_vector):
        def wrapper(p):
            for i in range(4):
                p = T.set_subtensor(p[i,null_id], zero_vector)
            return p
        return wrapper 

    def _compute_memories(self, statement, previous, weights, pe_matrix):
        
        '''
        print "weights"
        print weights
        print "pe_weights"
        print pe_weights
        '''
        #pe_weights = weights
        #memories = T.sum(pe_weights, axis=0)
        '''
        print "memories"
        print memories
        print "statement"
        print statement
        '''

        #pe_weights = pe_matrix * weights[statement]
        pe_weights = pe_matrix
        #pe_memories = T.sum(pe_weights, axis=0)
        #pe_memories = pe_matrix

        #a = T.matrix(dtype=theano.config.floatX)

        '''
        print "pe_matrix"
        print pe_matrix.shape
        print pe_matrix.dtype

        print "pe_weights"
        print pe_weights.shape
        print pe_weights.dtype

        print "pe_memories"
        print pe_memories.shape
        print pe_memories.dtype
        '''



        #memories = [statement]
        memories = statement

        #pe_memories = statement
        #memories = pe_memories
        #memories = statement
        #memories = pe_memories
        
        #memories = pe_matrix[]

        #memories = memories
        #memories.dtype = theano.config.floatX

        #memories = theano.shared(memories, dtype=theano.config.floatX)
        #memories = T.matrix(memories, dtype=theano.config.floatX)

        print "memories"
        #print memories.shape[0]
        #print memories.dtype

        
        return memories

    def _get_PE_matrix(self, num_words, embedding_size):
        pe_matrix = np.ones((num_words, 4, embedding_size), theano.config.floatX)
        #for j in range(num_words):
        #      for k in range(embedding_size):
        #          value = (1 - float(j+1)/num_words) - (float(k+1)/embedding_size) * (1 - 2*float(j+1)/num_words)
        #          for i in range(4):
        #              pe_matrix[j,i,k] = value
        #print "Here is pe_matrix"
        #print pe_matrix
        #model = skipthoughts.load_model()
        #vectors = skipthoughts.encode(model, X)              
        return pe_matrix

    def save_model(self, filename):
        f = file(filename, 'wb')
        for obj in [self.regularization, self.n_embedding, self.lr,
                    self.momentum, self.n_words, self.batch_size,
                    self.word_to_id, self.id_to_word, self.null_word_id,
                    self.max_stmts, self.max_words, self.weights, self.H]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load_model(self, filename):
        f = file(filename, 'rb')
        self.regularization = cPickle.load(f)
        self.n_embedding = cPickle.load(f)
        self.lr = cPickle.load(f)
        self.momentum = cPickle.load(f)
        self.n_words = cPickle.load(f)
        self.batch_size = cPickle.load(f)
        self.word_to_id = cPickle.load(f)
        self.id_to_word = cPickle.load(f)
        self.null_word_id = cPickle.load(f)
        self.max_stmts = cPickle.load(f)
        self.max_words = cPickle.load(f)
        self.weights = cPickle.load(f)
        self.H = cPickle.load(f)
        f.close()


    def memnn_batch_cost(self, statements_batch, question_batch, r_batch, pe_matrix):
        l = statements_batch.shape[0]
        #s, _ = theano.scan(fn=lambda i, c, xb, qb, rb, pe: T.as_tensor_variable(np.asarray(c - T.log(self.memnn_cost(xb[i], qb[i], pe)[rb[i]]), theano.config.floatX)),
        s, _ = theano.scan(#fn=lambda i, c, xb, qb, rb, pe: c - T.log(self.memnn_cost(xb[i], qb[i], pe)[rb[i]]),
                           fn=lambda i, c, xb, qb, rb, pe: c - T.log(self.memnn_cost(xb[i], qb[i], pe)[rb[i]]),
                           #outputs_info=self.X,
                           outputs_info=T.as_tensor_variable(np.asarray(0, theano.config.floatX)),
                           #outputs_info=T.ones_like(self.X),
                           non_sequences=[statements_batch, question_batch, r_batch, pe_matrix],
                           sequences=[theano.tensor.arange(l, dtype='int64')])
        #s = theano.shared(s, dtype=theano.config.floatX)
        return s[-1]

    def memnn_cost(self, statements, question, pe_matrix):
        # statements: list of list of word indices
        # question: list of word indices

        computed_memories, updates = theano.scan(
            self._compute_memories,
            sequences = statements,
            outputs_info = [
                #alloc_zeros_matrix(self.weights.shape[0])
                #alloc_zeros_matrix(self.weights.shape[0]),self.n_embedding) 
                alloc_zeros_matrix(self.weights.shape[0], 4800)   #init as 3
                #alloc_zeros_matrix(self.weights.shape[0], 4800, 4) #init as 4
                #alloc_zeros_matrix(4)
                #alloc_zeros_matrix(110, 4800)
            ],
            non_sequences = [
                #self.weights.dimshuffle(1, 0, 2),
                #self.weights.dimshuffle(1, 0, 2),
                self.weights,
                pe_matrix
            ],
            truncate_gradient = -1,
        )

        #memories = computed_memories
        #memories = T.stacklists(computed_memories)
        memories = T.stacklists(computed_memories).dimshuffle(1, 0, 2)
        #print computed_memories.shape[0]


        # Embed question
        #s = theano.tensor.scalar('s')
        #u1 = T.sum(self.weights[0][question], axis=0)
        #u1 = [question]
        u1 = question
        #u1 = u1.astype(np.float64)
        #u1 = np.asarray(u1, dtype=np.float64)
        #sv = skipthoughts.encode(model, sentence)


        # Layer 1
        p = T.nnet.softmax(T.dot(u1, memories[0].T))
        o1 = T.dot(p, memories[1])

        # Layer 2
        u2 = o1 + T.dot(u1, self.H)
        p = T.nnet.softmax(T.dot(u2, memories[1].T))
        o2 = T.dot(p, memories[2])

        # Layer 3
        u3 = o2 + T.dot(u2, self.H)
        p = T.nnet.softmax(T.dot(u3, memories[2].T))
        o3 = T.dot(p, memories[3])

        # Final
        output = T.nnet.softmax(T.dot(o3 + u3, self.weights[3].T))

        print "memnn_cost running"

        #return output[0, 1, 2, 3]
        return output[0]

    def train(self, dataset, questions, n_epochs=100, lr_schedule=None, start_epoch=0, max_words=4800):
        l_rate = self.lr
        index_array = np.arange(len(questions))

        # (max_words, )
        pe_matrix = self._get_PE_matrix(max_words, self.n_embedding)

        for epoch in xrange(start_epoch, start_epoch + n_epochs):
            costs = []

            if lr_schedule != None and epoch in lr_schedule:
                l_rate = lr_schedule[epoch]

            np.random.shuffle(index_array)
            seen = 0

            batches = make_batches(len(questions), self.batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                seen += len(batch_ids)
                questions_batch = []
                for index in batch_ids:
                    questions_batch.append(questions[index])

                '''
                print "length of questions"
                print len(questions)

                print "batch_start"
                print batch_start

                print "batch_end"
                print batch_end

                print "index"
                print index

                print "batch_ids"
                print batch_ids

                print "batch"
                print batches
                '''

                '''
                print "questions_batch inside"
                print questions_batch
                '''

                # (batch_size * max_stmts * max_words)
                #questions_batch = questions_batch.tolist()
                #statements_seq_batch = np.asarray(map(lambda x: x[2], questions_batch), theano.config.floatX)
                ##print "statements_seq_batch"
                statements_seq_batch = self.get_sub_element(questions_batch, 2)
                #statements_seq_batch = self.get_sub_element(questions_batch, 2, self.batch_size)
                
                statement_expand = np.expand_dims(statements_seq_batch, axis=0)

                ##print statements_seq_batch

                ##print "statement_expand"
                ##print statement_expand

                

                # (batch_size * max_words)
                #question_seq_batch = np.asarray(map(lambda x: x[3], questions_batch), theano.config.floatX)
                ##print "question_seq_batch"
                question_seq_batch = self.get_sub_element(questions_batch, 3)
                #question_seq_batch = self.get_sub_element(questions_batch, 3, self.batch_size)
                
                ##print question_seq_batch

                #print "question_seq_batch"
                #print question_seq_batch
                

                # (batch_size)
                #correct_word_batch = np.asarray(map(lambda x: x[4], questions_batch), theano.config.floatX)
                ##print "correct_word_batch"
                correct_word_batch = self.get_sub_element(questions_batch, 4)
                #correct_word_batch = self.get_sub_element(questions_batch, 4, self.batch_size)
                
                ##print correct_word_batch

                #print "correct_word_batch" 
                #print correct_word_batch
                
                '''
                print "length of questions"
                print len(questions)

                print "batch_start"
                print batch_start

                print "batch_end"
                print batch_end

                print "index"
                print index

                print "batch_ids"
                print batch_ids

                print "batch"
                print batches
                '''

                
                cost = self.train_function(
                        statement_expand,
                        question_seq_batch,
                        correct_word_batch,
                        pe_matrix,
                        l_rate
                    )
                

                '''
                cost = self.train_function(
                        statements_seq_batch,
                        question_seq_batch,
                        correct_word_batch,
                        pe_matrix,
                        l_rate
                    )
                '''

                #z = np.concatenate(([y],[a],[b]))

                '''
                statements_seq_batch = np.transpose(statements_seq_batch, (1, 0, 2))

                for i in range(len(correct_word_batch)):

                    statements_seq_batch_i = statements_seq_batch[i]   
                    question_seq_batch_i = question_seq_batch[i]
                    correct_word_batch_i = correct_word_batch[i]

                    cost = self.train_function(
                        statements_seq_batch_i,
                        question_seq_batch_i,
                        correct_word_batch_i,
                        pe_matrix,
                        l_rate
                    )
                '''

                #print "cost"
                #print cost

                # print "Epoch %d, sample %d: %f" % (epoch, i, cost)
                costs.append(cost)

            
            #print "questions_batch outside"
            #print questions_batch
            '''
            print "statements_seq_batch"
            print statements_seq_batch
            print "question_seq_batch"
            print question_seq_batch
            print "correct_word_batch" 
            print correct_word_batch
            '''

            #print "regularization_cost_out"
            #print self.regularization_cost

            #print "memory_loss_out"
            #print self.memory_loss

            print "costs"
            print costs
            print "Epoch %d: %f" % (epoch, np.mean(costs))

    def get_sub_element(self, x, i):
    #def get_sub_element(self, x, i, s):
    #for i in s:
        #y = [i-1][i]
        '''
        print "x"
        print x
        print "x[0]"
        print x[0]
        print "x[1]"
        print x[1]
        print "x[2]"
        print x[2]
        print "x[3]"
        print x[3]
        '''


        y0=x[0][i]
        y1=x[1][i]
        y2=x[2][i]
        y3=x[3][i]

        '''
        y20=x[0][2]
        y21=x[1][2]
        y22=x[2][2]
        y23=x[3][2]
        '''
        
        '''
        print "y0"
        print y0
        print "y1"
        print y1
        print "y2"
        print y2
        print "y3"
        print y3
        '''  

        '''
        ya0 = np.asarray(y20)
        print "ya0"
        print ya0
        print "ya0.shape"
        print ya0.shape 
        ya1 = np.asarray(y21)
        print "ya1"
        print ya1
        print "ya1.shape"
        print ya1.shape 
        ya2 = np.asarray(y22)
        print "ya2"
        print ya2
        print "ya2.shape"
        print ya2.shape 
        ya3 = np.asarray(y23)
        print "ya3"
        print ya3
        print "ya3.shape"
        print ya3.shape 
        '''
        
        y = [y0] + [y1] + [y2] + [y3]
        ##print "y"
        y = np.asarray(y, dtype=theano.config.floatX)
        ##print y.dtype
        #print y
        ##print y.shape
        #y = theano.shared(y, dtype=theano.config.floatX)
        #y = np.asarray(y)
        ##print "pass worked"
        ##print i
        return y


    def predict(self, dataset, questions, max_words=4800, print_errors=False):
        correct_answers = 0
        wrong_answers = 0
        pe_matrix = self._get_PE_matrix(max_words, self.n_embedding)

        for i, question in enumerate(questions):
            statements_seq = np.asarray(question[2], theano.config.floatX)
            question_seq = np.asarray(question[3], theano.config.floatX)
            correct = question[4]

            #statement_expand = np.expand_dims(statements_seq, axis=0)
            statement_expand = [statements_seq]

            
            probs = self.predict_function(
                statement_expand, question_seq, pe_matrix
            )

            '''
            probs = self.predict_function(
                statements_seq, question_seq, pe_matrix
            )
            '''
            
            predicted = np.argmax(probs)

            if len(question) == 6:
                ## For mc_test
                options = question[5]
                options_probs = probs[options]
                best_idx = np.argmax(options_probs)
                predicted = options[best_idx]
                ##

            if predicted == correct:
                correct_answers += 1
            else:
                if print_errors and np.random.rand() < 0.02:
                    print 'Correct: %s (%d %.3f), Guess: %s (%d %.3f)' % (self.id_to_word[correct], correct, probs[correct], self.id_to_word[predicted], predicted, probs[predicted])
                wrong_answers += 1

            if len(questions) > 1000:
                print '(%d/%d) %d correct, %d wrong' % (i+1, len(questions), correct_answers, wrong_answers)

        print '%d correct, %d wrong' % (correct_answers, wrong_answers)

if __name__ == "__main__":

    '''
    x = theano.tensor.dscalar('x')
    f = theano.function([x], [5 * x],
                        mode=theano.compile.MonitorMode(
                            pre_func=inspect_inputs,
                            post_func=inspect_outputs, detect_nan
                            ))
    f(3)
    '''    
    
    train_file = sys.argv[1]
    test_file = train_file.replace('train', 'test')

    #model = skipthoughts.load_model()
    #vectors = skipthoughts.encode(model, X)

    if len(sys.argv) > 2:
        n_epochs = int(sys.argv[2])
    else:
        n_epochs = 10

    if len(sys.argv) > 3:
        n_embedding = int(sys.argv[3])
    else:
        n_embedding = 80

    mode = 'babi' # babi or wiki

    if '.pickle' in train_file:
        mode = 'wiki'

    #MAKE SURE YOU UNCOMMENT THIS
    max_stmts = 110
    max_words = 4800
    

    if mode == 'babi':
        train_dataset, train_questions, word_to_id, num_words, null_word_id = parse_dataset_weak(train_file, max_stmts=max_stmts, max_words=max_words)
        test_dataset, test_questions, _, _, _ = parse_dataset_weak(test_file, word_id=num_words, word_to_id=word_to_id, update_word_ids=False, max_stmts=max_stmts, max_words=max_words)
    elif mode == 'wiki':
        # Check for pickled dataset
        print("Loading pickled train dataset")
        f = file(train_file, 'rb')
        import cPickle
        obj = cPickle.load(f)
        train_dataset, train_questions, word_to_id, num_words, null_word_id = obj

        print("Loading pickled test dataset")
        f = file(test_file, 'rb')
        obj = cPickle.load(f)
        test_dataset, test_questions, _, _, _ = obj
    elif mode == 'debug':
        train_dataset = []
        train_questions = [[0, 2, [[0, 1, 2, 3, 4, 5], [6, 7, 2, 3, 8, 5], [9, 10, 0, 11]], 4]]
        num_words = 12
        word_to_id = {}

    print "Dataset has %d words" % num_words
    # print train_questions[0]

    '''
    print "train_dataset"
    print train_dataset

    print "train_questions"
    print train_questions

    print "word_to_id"
    print word_to_id

    print "num_words"
    print num_words

    print "null_word_id"
    print null_word_id
    '''

   # model_file = "mctest500_dim100_wmemnn.pickle"
    train_my_model = True
    save_my_model = False



    if train_my_model:
        #wmemNN = WMemNN(n_words=num_words, n_embedding=100, lr=0.01, word_to_id=word_to_id, null_word_id=null_word_id,
        wmemNN = WMemNN(n_words=num_words, n_embedding=4800, lr=0.01, word_to_id=word_to_id, null_word_id=null_word_id,
                        max_stmts=max_stmts, max_words=max_words)

        lr_schedule = dict([(0, 0.01), (25, 0.01/2), (50, 0.01/4), (75, 0.01/8)])

        for i in xrange(n_epochs/2):

            wmemNN.train(train_dataset, train_questions, 5, lr_schedule, 5*i, max_words)
            wmemNN.predict(train_dataset, train_questions, max_words)
            wmemNN.predict(test_dataset, test_questions, max_words)

        if save_my_model:
            print "Saving model to", model_file
            wmemNN.save_model(model_file)
    else:
        wmemNN = WMemNN(load_from_file=model_file)
        wmemNN.predict(train_dataset, train_questions, max_words)
        wmemNN.predict(test_dataset, test_questions, max_words)


