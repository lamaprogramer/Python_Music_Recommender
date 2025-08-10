from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Sequential, Model

class Autoencoder(Model):
  """
  A wrapper class for containing encoder and decoder models.
  
  ## Constructor Args:
  
  **latent_dim**
  - Size of bottleneck to transform the data into.
  
  **shape**
  - Size of the input data
  """
  
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = Sequential([
      Input(shape=shape),
      Conv2D(32, (3, 3), activation='relu', padding='same'),
      MaxPooling2D((2, 2)),
      Conv2D(16, (3, 3), activation='relu', padding='same'),
      MaxPooling2D((2, 2)),
      Conv2D(latent_dim, (3, 3), activation='relu', padding='same'),
      MaxPooling2D((2, 2))
    ])
    
    self.decoder = Sequential([
      Input(shape=(int(shape[0]/2/2/2), int(shape[1]/2/2/2), latent_dim)),
      Conv2D(latent_dim, (3, 3), activation='relu', padding='same'),
      UpSampling2D((2, 2)),
      Conv2D(16, (3, 3), activation='relu', padding='same'),
      UpSampling2D((2, 2)),
      Conv2D(32, (3, 3), activation='relu', padding='same'),
      UpSampling2D((2, 2)),
      Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded