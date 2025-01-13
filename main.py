import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from io import StringIO
import tensorflow.keras.backend as K
from google.colab import drive
drive.mount('/content/drive')

# Function to get file names from a GitHub directory using the GitHub API
def get_github_file_urls(repo, path, branch="main"):
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        file_urls = []
        for file in files:
            if file['type'] == 'file' and file['name'].endswith('.csv'):
                # Construct the correct raw URL for each CSV file
                raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}/{file['name']}"
                file_urls.append(raw_url)
        return file_urls
    else:
        print(f"Failed to retrieve directory: {path} - Status code: {response.status_code}")
        return []

# Example: Get CSV file URLs from multiple GitHub repository directories
repo = "cedric-cnam/5G3E-dataset"
directories = [
    "SampleData/RAN_level/site1_2",
    "SampleData/RAN_level/site3",  # Another directory
    "SampleData/RAN_level/site4",
]

# List to store all CSV file URLs
all_csv_file_urls = []

# Loop through each directory and fetch CSV file URLs
for directory in directories:
    print(f"Fetching CSV files from directory: {directory}")
    csv_file_urls = get_github_file_urls(repo, directory)
    if csv_file_urls:
        all_csv_file_urls.extend(csv_file_urls)
    else:
        print(f"No CSV files found in directory: {directory}")

# Now, let's download all the CSV files and inspect them before concatenating
all_dataframes = []

# Loop through the CSV URLs and download the files
for url in all_csv_file_urls:
    print(f"Downloading file from: {url}")
    try:
        # Download CSV content and read into pandas DataFrame, ensuring first row is used as header and delimiter is semicolon
        df = pd.read_csv(url, delimiter=';', header=0)  # Ensure the first row is treated as column headers
        # Check the first few rows to ensure it's loaded correctly
        print("DataFrame loaded:")
        print(df.head())  # Print first few rows of the DataFrame to verify the content
        all_dataframes.append(df)
    except Exception as e:
        print(f"Error reading CSV from {url}: {e}")

# Check if there are DataFrames to concatenate
if all_dataframes:
    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    # Display the head of the combined DataFrame
    print("Combined DataFrame:")
    print(combined_df.head())
else:
    print("No data frames to concatenate.")

# Step 1: Check for missing data and handle it (if any)
print("Checking for missing data:")
print(combined_df.isnull().sum())  # Check for missing values

# You can choose to fill missing values or drop them
combined_df = combined_df.fillna(0)  # Example: Filling missing values with 0

# Step 2: Normalize the numerical features (scaling)
# You can use MinMaxScaler or StandardScaler depending on your needs
scaler = MinMaxScaler()
scaled_columns = ['time', 'nof_ue', 'dl_brate', 'ul_brate', 'proc_rmem', 'proc_rmem_kB',
                  'proc_vmem_kB', 'sys_mem', 'system_load', 'thread_count']  # Add more columns as needed

# Ensure these columns exist in the dataframe
scaled_columns = [col for col in scaled_columns if col in combined_df.columns]

# Normalize selected columns
combined_df[scaled_columns] = scaler.fit_transform(combined_df[scaled_columns])

# Step 3: Prepare the data for training
X = combined_df.values  # All the features (input data for the CT-GAN)

# Step 4: Prepare the TensorFlow Dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(combined_df)

# Convert the data to tensorflow Dataset
dataset = tf.data.Dataset.from_tensor_slices(data_scaled)
batch_size = 64  # You can change this based on your available memory
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

# Check the dataset
for data in dataset.take(1):
    print(data)

# Generator Model
def build_generator(z_dim, condition_dim):
    z = layers.Input(shape=(z_dim,))
    condition = layers.Input(shape=(condition_dim,))

    x = layers.Concatenate()([z, condition])
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    out = layers.Dense(condition_dim, activation='tanh')(x)

    generator = Model([z, condition], out)
    return generator

z_dim = 100
condition_dim = data_scaled.shape[1]
generator = build_generator(z_dim, condition_dim)
generator.summary()

# Discriminator Model
def build_discriminator(condition_dim):
    data = layers.Input(shape=(condition_dim,))
    condition = layers.Input(shape=(condition_dim,))

    x = layers.Concatenate()([data, condition])
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    discriminator = Model([data, condition], out)
    return discriminator

discriminator = build_discriminator(condition_dim)
discriminator.summary()

# Optimizers and Loss Function
lr = 0.0002
g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Training Loop
epochs = 100
for epoch in range(epochs):
    for batch_data in dataset:
        batch_size = batch_data.shape[0]
        z = tf.random.normal([batch_size, z_dim])
        condition = batch_data

        real_labels = tf.ones((batch_size, 1))  # Real data label = 1
        fake_labels = tf.zeros((batch_size, 1))  # Fake data label = 0

        # Train Discriminator
        with tf.GradientTape() as tape_d:
            real_output = discriminator([batch_data, condition], training=True)
            d_loss_real = cross_entropy(real_labels, real_output)

            fake_data = generator([z, condition], training=False)
            fake_output = discriminator([fake_data, condition], training=True)
            d_loss_fake = cross_entropy(fake_labels, fake_output)

            d_loss = d_loss_real + d_loss_fake

        grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as tape_g:
            fake_data = generator([z, condition], training=True)
            fake_output = discriminator([fake_data, condition], training=True)
            g_loss = cross_entropy(real_labels, fake_output)

        grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.numpy()}, G Loss: {g_loss.numpy()}")

# Save models locally in Colab
generator.save('/content/generator.h5')
discriminator.save('/content/discriminator.h5')

print("Models saved locally in Colab.")
