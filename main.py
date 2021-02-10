from model import FashionMnist

if __name__ == '__main__':
    mnist = FashionMnist('fashion_mnist')
    mnist.build()
    mnist.train()

    print('\n--------Test--------\n')
    x, label = mnist.data.test.next_batch(100)
    mnist.eval(x, label)
    mnist.close_sess()

