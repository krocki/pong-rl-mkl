from __future__ import print_function, division
import ctypes
from multiprocessing import Process, Array
import datetime, time
import pickle as pickle
import numpy as np
## Create the dictionary mapping ctypes to numpy dtypes.
ctype2dtype = {}
n_procs=4

## Integer types
for prefix in ( 'int', 'uint' ):
    for log_bytes in range( 4 ):
        ctype = '%s%d_t' % ( prefix, 8*(2**log_bytes) )
        dtype = '%s%d' % ( prefix[0], 2**log_bytes )
        # print( ctype )
        # print( dtype )
        ctype2dtype[ ctype ] = np.dtype( dtype )

## Floating point types
ctype2dtype[ 'float' ] = np.dtype( 'f4' )
ctype2dtype[ 'double' ] = np.dtype( 'f8' )

# print( ctype2dtype )

def asarray( ffi, ptr, length ):
    ## Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype( ffi.typeof( ptr ).item )
    # print( T )
    # print( ffi.sizeof( T ) )

    if T not in ctype2dtype:
        raise RuntimeError( "Cannot create an array for element type: %s" % T )

    return np.frombuffer( ffi.buffer( ptr, length*ffi.sizeof( T ) ), ctype2dtype[T] )

def init_mem(ffi, D, H, A):

  MAX_EP_LEN = 20000

  c_model, p_model, c_grad, p_grad, c_mem, p_mem = {}, {}, {}, {}, {}, {}
  c_model['W1'] = ffi.new( "float[]", H * D )
  c_model['W2'] = ffi.new( "float[]", H * A )
  c_grad['W1'] = ffi.new( "float[]", H * D )
  c_grad['W2'] = ffi.new( "float[]", H * A )
  c_mem['W1'] = ffi.new( "float[]", H * D )
  c_mem['W2'] = ffi.new( "float[]", H * A )

  p_model['W1'] = asarray( ffi, c_model['W1'], H * D )
  p_model['W2'] = asarray( ffi, c_model['W2'], H * A )
  p_grad['W1'] = asarray( ffi, c_grad['W1'], H * D )
  p_grad['W2'] = asarray( ffi, c_grad['W2'], H * A )
  p_mem['W1'] = asarray( ffi, c_mem['W1'], H * D )
  p_mem['W2'] = asarray( ffi, c_mem['W2'], H * A )

  c_x = ffi.new( "float[]", D); p_x = asarray(ffi, c_x, D)
  c_h = ffi.new( "float[]", H); p_h = asarray(ffi, c_h, H)
  c_dh = ffi.new( "float[]", H*MAX_EP_LEN); p_dh = asarray(ffi, c_dh, H*MAX_EP_LEN)
  c_logp = ffi.new( "float[]", A); p_logp = asarray(ffi, c_logp, A)
  c_p = ffi.new( "float[]", A); p_p = asarray(ffi, c_p, A)

  c_epx = ffi.new( "float[]", D * MAX_EP_LEN); p_epx = asarray(ffi, c_epx, D * MAX_EP_LEN)
  c_eph = ffi.new( "float[]", H * MAX_EP_LEN); p_eph = asarray(ffi, c_eph, H * MAX_EP_LEN)
  c_epdlogp = ffi.new( "float[]", A * MAX_EP_LEN); p_epdlogp = asarray(ffi, c_epdlogp, A * MAX_EP_LEN)
  c_epr = ffi.new( "float[]", 1 * MAX_EP_LEN); p_epr = asarray(ffi, c_epr, 1 * MAX_EP_LEN)

  return c_model, p_model, c_grad, p_grad, c_mem, p_mem, c_epx, p_epx, c_eph, p_eph, c_epdlogp, p_epdlogp, c_epr, p_epr, c_x, p_x, c_dh, c_h, p_dh, p_h, c_logp, p_logp, c_p, p_p

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def printlog(name,s):
    print(s) ; f=open(name + '_log.txt','a') ; f.write(s+'\n') ; f.close()

def v2bar(v):
  bar = ''.join(['*'] * int(v)) + ''.join(['.'] * int(42-v))
  return bar

def train(_arr1, _arr2, p_model, idx, name):

  from cffi import FFI
  ffi = FFI()

  ffi.cdef("""
           void print_w(float*, size_t N);
           void forward(float* x, float* h, float* logp, float* p, float* w1, float *w2);
           void copy1(float* dst, unsigned int idx, float val);
           void copyn(float* dst, unsigned int idx, float* src, unsigned int n);
           void modulate(float *logp, float* r, float gamma, unsigned int len);
           void adapt(float *w, float *m, float *g, unsigned int n, float lr, float decay);
           void backward(float* eph, float* epdlogp, float *epx, float *dW1, float* dW2, float *dh, float* W2, int ep_length);
           """)

  C = ffi.dlopen("./nn.so")
  H = 100
  D = 80 * 80
  A = 1
  gamma = 0.99
  learning_rate = 5*1e-4
  decay = 0.99

  resume = False
  render = False

  _, _, c_grad, p_grad, c_mem, p_mem, c_epx, p_epx, c_eph, p_eph, c_epdlogp, p_epdlogp, c_epr, p_epr, c_x, p_x, c_dh, c_h, p_dh, p_h, c_logp, p_logp, c_p, p_p = init_mem(ffi, D, H, A)

  max_l = 0

  arr1 = ffi.cast('float*', _arr1.ctypes.data)
  arr2 = ffi.cast('float*', _arr2.ctypes.data)

  p_mem['W1'][:] = np.zeros((H,D)).flatten()
  p_mem['W2'][:] = np.zeros((A,H)).flatten()

  env = gym.make("Pong-v4")
  observation = env.reset()
  prev_x = None # used in computing the difference frame

  it, episodes, reward_sum, running_reward = 0, 0, 0, -21
  xs, hs, dlogps, drs = [], [], [], []

  while True:

    if render: env.render()
    it += 1
    cur_x = prepro(observation)
    p_x[:] = cur_x - prev_x if prev_x is not None else np.zeros((D,1)).flatten()
    prev_x = cur_x

    C.forward(c_x, c_h, c_logp, c_p, arr1, arr2)
    # store x and h
    C.copyn(c_epx, (it-1)*D, c_x, D); C.copyn(c_eph, (it-1)*H, c_h, H);

    # action, step, get dy
    action = 2 if np.random.uniform() < p_p else 3 # roll the dice!
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    y = 1 if action == 2 else 0 # a "fake label"
    p_logp[:] = y - p_p

    # store prob, logp
    C.copyn(c_epdlogp, (it-1)*A, c_logp, A)
    C.copy1(c_epr, (it-1), reward)

    if done:
      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

      # backward + adapt
      C.modulate(c_epdlogp, c_epr, gamma, it);
      C.backward(c_eph, c_epdlogp, c_epx, c_grad['W1'], c_grad['W2'], c_dh, arr2, it)
      C.adapt(arr1, c_mem['W1'], c_grad['W1'], H*D, learning_rate, decay)
      C.adapt(arr2, c_mem['W2'], c_grad['W2'], H*A, learning_rate, decay)

      if it > max_l: max_l = it

      printlog('raw_'+name, '{:5d} {:8.2f} {:5d} {:5d} {:5d} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:20s}'.format( idx, time.time()-t_start, episodes, it, max_l, np.linalg.norm(p_model['W1']), np.linalg.norm(p_model['W2']), np.linalg.norm(p_grad['W1']), np.linalg.norm(p_grad['W2']), np.linalg.norm(p_mem['W1']), np.linalg.norm(p_mem['W2']), np.linalg.norm(p_epdlogp[:it]), reward_sum, running_reward, v2bar(21+reward_sum)))
      #printlog(name, '{:10s} t {:5.1f} e {:5d} l {:5d} max {:5d} n1 {:6.1f} n2 {:6.2f} g1 {:6.2f} g2 {:6.2f} m1 {:6.2f} m2 {:6.2f} e {:6.2f}   r {:5.1f} avg {:6.2f} {:20s}'.format( name, time.time()-t_start, episodes, it, max_l, np.linalg.norm(p_model['W1']), np.linalg.norm(p_model['W2']), np.linalg.norm(p_grad['W1']), np.linalg.norm(p_grad['W2']), np.linalg.norm(p_mem['W1']), np.linalg.norm(p_mem['W2']), np.linalg.norm(p_epdlogp[:it]), reward_sum, running_reward, v2bar(21+reward_sum)))

      it = 0
      xs, hs, dlogps, drs = [], [], [], []
      observation = env.reset() # reset env
      prev_x = None
      reward_sum = 0
      episodes += 1

      if episodes % 100 == 0: pickle.dump(p_model, open('save.p', 'wb'))

import gym
from cffi import FFI
t_start=time.time()

if __name__ == '__main__':

  ffi = FFI()

  ffi.cdef("""
           void print_w(float*, size_t N);
           void forward(float* x, float* h, float* logp, float* p, float* w1, float *w2);
           void copy1(float* dst, unsigned int idx, float val);
           void copyn(float* dst, unsigned int idx, float* src, unsigned int n);
           void modulate(float *logp, float* r, float gamma, unsigned int len);
           void adapt(float *w, float *m, float *g, unsigned int n, float lr, float decay);
           void backward(float* eph, float* epdlogp, float *epx, float *dW1, float* dW2, float *dh, float* W2, int ep_length);
           """)

  C = ffi.dlopen("./nn.so")

  H = 200
  D = 80 * 80
  A = 1
  gamma = 0.99
  learning_rate = 5*1e-4
  decay = 0.99

  resume = False
  render = False


  c_model, p_model, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = init_mem(ffi, D, H, A)
  shared_w1 = Array(ctypes.c_float, H*D, lock=False)
  shared_w2 = Array(ctypes.c_float, H*A, lock=False)
  shared_w1_p = np.ctypeslib.as_array(shared_w1)
  shared_w2_p = np.ctypeslib.as_array(shared_w2)

  c_model['W1'] = shared_w1
  p_model['W1'] = shared_w1_p
  c_model['W2'] = shared_w2
  p_model['W2'] = shared_w2_p

  arr1 = c_model['W1']
  arr2 = c_model['W2']

  if resume:
    p_model = pickle.load(open(name + 'save.p', 'rb'))
  else:
    p_model['W1'][:] = (np.random.randn(H,D) / np.sqrt(D)).flatten()
    p_model['W1'] = np.reshape(p_model['W1'], (H,D))
    p_model['W2'][:] = (np.random.randn(H,A) / np.sqrt(H)).flatten()
    p_model['W2'] = np.reshape(p_model['W2'], (A,H))
  ids = range(n_procs)
  procs = []
  for i in ids:
    proc = Process(target=train, args=(p_model['W1'], p_model['W2'], p_model, i, 'proc_' + str(i),))
    procs.append(proc)
    proc.start()

  for proc in procs:
      proc.join()

