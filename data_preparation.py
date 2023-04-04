import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
import nltk 
import string
import re
import gc 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud
import imgaug.augmenters as iaa #data augmentation
import cv2 #image processing library

#define preprocess function
def preprocess(text):
    pattern = re.compile(r'[\[\]|@"#&~\\\']') # pattern to remove special characters
    text = re.sub(pattern, '', text) 
    text = text.lower() # Convert to lowercase
    return text 
def display_3_images(im1, im2, im3):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax3.imshow(im3)
    st.pyplot(fig)
# Define the Streamlit app
def app():
    # Set the app title and favicon
    #st.set_page_config(page_title='Word Problem to Image Converter', page_icon=':pencil2:')

    # Define the project title and subtitle
    st.title("Word Problem to Image Converter")
    st.subheader("Transform text-based math problems into visual images")
    # Define the project logo
    team_logo = Image.open("streamlit_template/logo.png")
    project_logo = Image.open("streamlit_template/logo2.png")
    st.image(project_logo, use_column_width=True)
    # Add a section for the project description
    st.markdown("---")
    st.header("Project Description")
    st.write("This project aims to convert text-based math problems into visual images that can be easily understood by students. By using natural language processing techniques and computer vision algorithms, we can transform complex word problems into simple diagrams and illustrations that make learning math more engaging and intuitive.")

    # Add a section for the input data
    st.markdown("---")
    st.header("Input Data")
    st.write("The input to this project consists of text-based math problems in natural language format. These problems can be sourced from various textbooks, websites, or other educational materials.")
    st.markdown("#### Examples :")
    word_problem1 = Image.open('streamlit_template/word_problem1.png')
    st.image(word_problem1)
    # add a space 
    st.text("      ")
    word_problem2 = Image.open('streamlit_template/word_problem2.png')
    st.image(word_problem2)
    # Add a section for the output data
    st.markdown("---")
    st.header("Output Data")
    st.write("The output of this project is a set of visual images that represent the solutions to the original math problems. These images may consist of diagrams, graphs, or other visual representations that help students understand the problem and its solution.")
    st.markdown("#### Examples :")
    # Load the images
    image1 = Image.open('streamlit_template/image1.png')
    image2 = Image.open('streamlit_template/image2.png')
    image3 = Image.open('streamlit_template/image3.png')
    image4 = Image.open('streamlit_template/image4.png')
    # Set width and padding for the divs
    image_height = 150

# Resize the images to the desired height
    image1 = image1.resize((int(image_height*image1.width/image1.height), image_height))
    image2 = image2.resize((int(image_height*image2.width/image2.height), image_height))
    image3 = image3.resize((int(image_height*image3.width/image3.height), image_height))
    image4 = image4.resize((int(image_height*image4.width/image4.height), image_height))
     # Display the images in a horizontal layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image1, width=150)
        st.text("      ")
    with col2:
        st.image(image2, width=150)
        st.text("      ")
    with col3:
        st.image(image3, width=150)
        st.text("      ")

    with col4:
        st.image(image4, width=150) 
        
# Display the images in a horizontal layout
    #st.image([image1, image2, image3, image4], width=250)
    # Add a section for the technologies used
    st.markdown("---")
    st.header("Technologies Used")
    st.write("This project uses natural language processing techniques and computer vision algorithms to transform text-based math problems into visual images. Specifically, we use Python and the following libraries: spaCy, OpenCV, Matplotlib, and PIL.")

    # Add a section for the potential impact
    st.markdown("---")
    st.header("Potential Impact")
    st.write("This project has the potential to revolutionize the way math is taught in schools by making it more engaging and accessible for students of all ages and skill levels. By converting complex math problems into simple visual representations, we can help students develop a deeper understanding of mathematical concepts and improve their overall performance in the subject.")

    # Add a footer with the project logo and links to the Github repository and author's website
    st.markdown("---")
    st.image(team_logo, use_column_width=True)
    st.write("Created by [Neural network nagivators](#) | [Github Repository](https://github.com/RoukaiaKHELIFI/Educational-Interactive-Intelligent-Platform)")
    st.write("Logo design by [Roukaia Khelifi](#)")

# Call the Streamlit app
# Define the pages in your app
def home():
    st.markdown("# Word Problems & Levels")

    # # Data Cleaning and Preparation
    # # TODO: Add code here to clean and prepare data as needed

    # Set the maximum number of datasets to upload
    MAX_DATASETS = 4

# Use a slider to let the user choose the number of datasets to upload
    num_datasets = st.slider("Choose the number of datasets to upload", 1, MAX_DATASETS)

    # Display the file uploaders only if the user selects the correct number of datasets
    if num_datasets == 1:
        uploaded_file1 = st.file_uploader("Upload the first file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        if uploaded_file1 is not None:
            # Read the uploaded file
            df1 = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith('.csv') else pd.read_excel(uploaded_file1)
            st.write(f"Preview of dataset 1:")
            st.write(df1.head())

    if num_datasets == 2:
        uploaded_file1 = st.file_uploader("Upload the first file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        uploaded_file2 = st.file_uploader("Upload the second file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        if uploaded_file1 is not None and uploaded_file2 is not None:
            # Read the uploaded files
            df1 = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith('.csv') else pd.read_excel(uploaded_file1)
            df2 = pd.read_csv(uploaded_file2) if uploaded_file2.name.endswith('.csv') else pd.read_excel(uploaded_file2)
            st.write(f"Preview of dataset 1:")
            st.write(df1.head())
            st.write(f"Preview of dataset 2:")
            st.write(df2.head())

    if num_datasets == 3:
        uploaded_file1 = st.file_uploader("Upload the first file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        uploaded_file2 = st.file_uploader("Upload the second file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        uploaded_file3 = st.file_uploader("Upload the third file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        if uploaded_file1 is not None and uploaded_file2 is not None and uploaded_file3 is not None:
            # Read the uploaded files
            df1 = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith('.csv') else pd.read_excel(uploaded_file1)
            df2 = pd.read_csv(uploaded_file2) if uploaded_file2.name.endswith('.csv') else pd.read_excel(uploaded_file2)
            df3 = pd.read_csv(uploaded_file3) if uploaded_file3.name.endswith('.csv') else pd.read_excel(uploaded_file3)
            st.write(f"Preview of dataset 1:")
            st.write(df1.head())
            st.write(f"Preview of dataset 2:")
            st.write(df2.head())
            st.write(f"Preview of dataset 3:")
            st.write(df3.head())


            K = st.slider("Select a value for K:",min_value=0,max_value=2,value=1)
            if K ==2:
                df1['Problem'] = df1['Problem'].apply(preprocess)
                df2['Problem'] = df2['Problem'].apply(preprocess)
                df3['Problem'] = df3['Problem'].apply(preprocess)

                # Create TF-IDF vectorizer
                vectorizer1 = TfidfVectorizer()
                vectorizer2 = TfidfVectorizer()
                vectorizer3 = TfidfVectorizer()
                X1 = vectorizer1.fit_transform(df1['Problem'])
                X2 = vectorizer2.fit_transform(df2['Problem'])
                X3 = vectorizer3.fit_transform(df3['Problem'])

                # Cluster the word problems using K-Means
                kmeans1 = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=1)
                kmeans1.fit(X1)
                y_kmeans1 = kmeans1.predict(X1)
                df1['cluster'] = y_kmeans1
                df1.loc[df1['cluster'] == 0, 'level'] = 'Medium'
                df1.loc[df1['cluster'] == 1, 'level'] = 'Hard'
                df1.drop(['options','category','cluster'],axis=1,inplace=True)
                df1['level'] = df1['level'].map({'Medium':2,'Hard':3})


                kmeans2 = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=1)
                kmeans2.fit(X2)
                y_kmeans2 = kmeans2.predict(X2)
                df2['cluster'] = y_kmeans2
                df2.loc[df2['cluster'] == 0, 'level'] = 'Medium'
                df2.loc[df2['cluster'] == 1, 'level'] = 'Hard'
                df2.drop(['options','category','cluster'],axis=1,inplace=True)
                df2['level'] = df2['level'].map({'Medium':2,'Hard':3})


                kmeans3 = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=1)
                kmeans3.fit(X3) 
                y_kmeans3 = kmeans3.predict(X3)
                df3['cluster'] = y_kmeans3
                df3.loc[df3['cluster'] == 0, 'level'] = 'Medium'
                df3.loc[df3['cluster'] == 1, 'level'] = 'Hard'
                df3.drop(['options','category','cluster'],axis=1,inplace=True)
                df3['level'] = df3['level'].map({'Medium':2,'Hard':3})

                #combine the dataframes
                df = pd.concat([df1,df2,df3],axis=0)
                df = df.rename(columns={'Problem': 'question'})
                st.write(f"Preview of the final dataset :")
                st.write(df.head())

                #uploading level 1 & level 2 dataset
                dataset2 = pd.read_csv('C:/DsProjects/ScrapingProject/dataset2.csv')
                dataset2['question'] = dataset2['question'].apply(preprocess)
                #combining the dataframes
                en_df = pd.concat([df,dataset2],axis=0)

                #shape of the dataset
                st.markdown("## Shape of the final dataset :")
                st.write(en_df.shape)

                #data visualization
                
                # count the number of problems per level and plot as a bar chart
                problem_counts = en_df['level'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=problem_counts.index, y=problem_counts, ax=ax, palette='viridis')
                ax.set_title('Number of Problems by Level', fontsize=18, fontweight='bold')
                ax.set_xlabel('Level', fontsize=14)
                ax.set_ylabel('Count', fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
                st.pyplot(fig)


                # Compute problem lengths
                en_df['problem_length'] = en_df['question'].apply(lambda x: len(x))
                st.markdown("## A histogram of problem lengths")
                # Plot histogram of problem lengths
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data=en_df, x='problem_length', bins=20, ax=ax, kde=False, color='#1f77b4')
                ax.set_title('Distribution of Problem Lengths', fontsize=18, fontweight='bold')
                ax.set_xlabel('Problem Length', fontsize=14)
                ax.set_ylabel('Count', fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
                st.pyplot(fig)
                st.markdown("## A scatter plot of problem lengths and frequencies")
                # Calculate problem frequencies
                en_df['problem_freq'] = en_df.groupby('question')['question'].transform('count')

                # Create scatter plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=en_df, x='problem_length', y='problem_freq', ax=ax, color='darkblue', alpha=0.5)
                ax.set_title('Problem Length vs Frequency', fontsize=18, fontweight='bold')
                ax.set_xlabel('Problem Length', fontsize=14)
                ax.set_ylabel('Problem Frequency', fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
                st.pyplot(fig)

                st.markdown("## problem lengths by level")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x='level', y='problem_length', data=en_df, ax=ax, palette='viridis')
                ax.set_title('Problem Lengths by Level', fontsize=18, fontweight='bold')
                ax.set_xlabel('Level', fontsize=14)
                ax.set_ylabel('Problem Length', fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
                st.pyplot(fig)

                st.markdown("## plot a pie chart of problem difficulty levels")
                level_counts = en_df['level'].value_counts()
                labels = level_counts.index.tolist()
                sizes = level_counts.tolist()

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, counterclock=False, colors=sns.color_palette('viridis', len(labels)))
                ax.set_title('Problem Difficulty Levels', fontsize=18, fontweight='bold')
                ax.axis('equal')
                ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
                st.pyplot(fig)

                st.markdown("## plot a word cloud of the most common words in the problems")
                # Create a word cloud from the "question" column of your DataFrame
                wordcloud = WordCloud(width=800, height=400).generate(" ".join(df["question"]))

                # Display the word cloud using Streamlit's st.pyplot() function
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_axis_off()
                st.pyplot(fig)


                st.markdown("## Split the problems into different levels")
              
                level1 = en_df[en_df['level'] == 1]
                level2 = en_df[en_df['level'] == 2]
                level3 = en_df[en_df['level'] == 3]

                # Function to generate wordcloud
                def generate_wordcloud(data, level):
                    text = " ".join(problem for problem in data.question)
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title("Wordcloud for Level " + str(level), fontsize=20)
                    st.pyplot(fig)

                # Generate wordcloud for each level
                generate_wordcloud(level1, 1)
                generate_wordcloud(level2, 2)
                generate_wordcloud(level3, 3)




    if num_datasets == 4:
        uploaded_file1 = st.file_uploader("Upload the first file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        uploaded_file2 = st.file_uploader("Upload the second file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        uploaded_file3 = st.file_uploader("Upload the third file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        uploaded_file4 = st.file_uploader("Upload the third file (CSV or Excel)", type=["csv", "xlsx", "xls"])
        if uploaded_file1 is not None and uploaded_file2 is not None and uploaded_file3 is not None and uploaded_file4 is not None:
            # Read the uploaded files
            df1 = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith('.csv') else pd.read_excel(uploaded_file1)
            df2 = pd.read_csv(uploaded_file2) if uploaded_file2.name.endswith('.csv') else pd.read_excel(uploaded_file2)
            df3 = pd.read_csv(uploaded_file3) if uploaded_file3.name.endswith('.csv') else pd.read_excel(uploaded_file3)
            df4 = pd.read_csv(uploaded_file4) if uploaded_file4.name.endswith('.csv') else pd.read_excel(uploaded_file4)
            st.write(f"Preview of dataset 1:")
            st.write(df1.head())
            st.write(f"Preview of dataset 2:")
            st.write(df2.head())
            st.write(f"Preview of dataset 3:")
            st.write(df3.head())
            st.write(f"Preview of dataset 4:")
            st.write(df4.head())
   


        

        # Add the cluster labels to the dataframe
        

        # Manually label the clusters as representing the three levels of complexity
        
            # Data Visualization
    # TODO: Add code here to create data visualizations using matplotlib, seaborn or other libraries

    # Show Data
      

def data_prep():
    st.write("Here is where you can prepare your data.")

def data_viz():
    st.write("Here is where you can visualize your data.")

def image_prep():
    st.title("Image Processing ")
    img = cv2.imread('img_preprocessing/Ben eats 6 strawberries at breakfast and 3 strawberries at lunch. How many strawberries did he eat altogether.PNG')
    img1 = cv2.imread('img_preprocessing/There are 6 birds in a tree and 7 birds are in the next tree. How many birds are in the tree altogether.PNG')
    img2 = cv2.imread('img_preprocessing/I buy 6 bottles of lemonade .If there are 2 litres in each bottle, how many litres of lemonade have I bought.PNG')
    st.markdown("## Dataset Overview :")
    #display images
    st.image(Image.open('streamlit_template/excel_ss.png'), width=700)



    st.markdown("## Original Images :")
    display_3_images(img, img1, img2)   
    
    st.markdown("## Resized Image (512x512) :")
    img_resized = cv2.resize(img, (512, 512))
    img_resized1 = cv2.resize(img1, (512, 512))
    img_resized2 = cv2.resize(img2, (512, 512))
    display_3_images(img_resized, img_resized1, img_resized2)

    st.markdown("## Normalizing the images :")
    st.success("We normalized the pixel values of the images to have zero mean and unit variance, which can improve model performance.")
    st.info(" - alpha is a parameter that scales the pixel values of the input image. This means that each pixel value in the input image is multiplied by the alpha value.\n\n- beta is a parameter that shifts the pixel values of the input image. This means that each pixel value in the input image is added to the beta value.")
    img_normalized = cv2.normalize(img_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_normalized1 = cv2.normalize(img_resized1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_normalized2 = cv2.normalize(img_resized2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    st.markdown("## Augmenting the images:")  
    st.info("Data augmentation can improve the performance of the model by creating new training samples. Use libraries like imgaug or albumentations to augment the images")

    st.markdown("## Rotating the images:")
    st.info("Image Rotation can help to reduce overfitting by increasing the diversity of the training data. By creating multiple versions of each image with different rotation angles")
    augmenter = iaa.Sequential([iaa.Flipud(0.5), iaa.Affine(rotate=(-10, 10))])

    img_augmented = augmenter(image=img_normalized)

    img_augmented1 = augmenter(image=img_normalized1)

    img_augmented2 = augmenter(image=img_normalized2)

    display_3_images(img_augmented,img_augmented1,img_augmented2)

    st.markdown("## Blurring:")
    st.info("Image blurring or smoothing can be done using various filter kernels to remove noise and reduce image details")

    img_blur = cv2.blur(img_normalized, (5, 5))
    img_blur1 = cv2.blur(img_normalized1, (5, 5))
    img_blur2 = cv2.blur(img_normalized2, (5, 5))
    display_3_images(img_blur,img_blur1,img_blur2)

    

    st.markdown("## Grayscale Conversion")
    st.info("Grayscale conversion can simplify the image processing and reduces the amount of data required for analysis")
    img_gray_scale = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray_scale1 = cv2.cvtColor(img_resized1, cv2.COLOR_BGR2GRAY)
    img_gray_scale2 = cv2.cvtColor(img_resized2, cv2.COLOR_BGR2GRAY)
    display_3_images(img_gray_scale,img_gray_scale1,img_gray_scale2)

    st.markdown("## Image Edge Detection")
    st.info("Image edge detection is a process of identifying the boundaries between regions with different color intensities in an image")

    img_edges = cv2.Canny(img_gray_scale, 100, 200)
    img_edges1 = cv2.Canny(img_gray_scale1, 100, 200)
    img_edges2 = cv2.Canny(img_gray_scale2, 100, 200)
    display_3_images(img_edges,img_edges1,img_edges2)
    st.markdown("## Image Thresholding")
    st.info("Image thresholding is a process of converting a grayscale image into a binary image by setting all pixels above a certain threshold to white and all below to black")
    _, img_thresh = cv2.threshold(img_gray_scale, 128, 255, cv2.THRESH_BINARY)
    _, img_thresh1 = cv2.threshold(img_gray_scale1, 128, 255, cv2.THRESH_BINARY)
    _, img_thresh2 = cv2.threshold(img_gray_scale2, 128, 255, cv2.THRESH_BINARY)
    display_3_images(img_thresh,img_thresh1,img_thresh2)




# Navigation bar
pages = {
        "Home":app,
        "Problems & Levels": home,
        "Image Processing": image_prep,
        "Output Data": data_viz,
    }
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .sidebar .sidebar-content {
        transition: margin-left .3s, margin-right .3s;
        z-index: 999999;
        position: absolute;
        width: 200px;
        height: 100%;
        overflow-y: auto;
        top: 52px;
        left: -200px;
        background-color: var(--secondary-background-color);
        border-right: 1px solid var(--light-primary-color);
    }
    .sidebar:hover .sidebar-content {
        margin-left: 0;
        margin-right: 0;
        transition: margin-left .3s, margin-right .3s;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.expander("Navigation"):
    page = st.sidebar.selectbox("Select a page", list(pages.keys()))
if page == "Home":
    st.empty()
    pages["Home"]()
elif page =="Problems & Levels":
    st.empty()
    pages["Problems & Levels"]()
elif page == "Output Data":
    st.empty()
    pages["Image Processing"]()
else :
    st.empty()
    pages["Image Processing"]()















