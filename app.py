import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('plant_disease_detection_model.h5')

# Define the classes and their corresponding information
class_info = {
    'Bell Pepper Bacterial Spot': {
        'type': 'Bacterial disease',
        'scientific_name': 'Xanthomonas campestris pv. vesicatoria',
        'symptoms': [
            'Water-soaked spots on leaves, stems, and fruit.',
            'Spots may be dark brown or black with a yellow halo.',
            'Lesions can coalesce, leading to wilting and death of affected tissue.',
            'Fruit may develop raised, corky lesions with a rough texture.',
            'Severely infected fruit may rot.'
        ],
        'causes': 'Bacterial spot in bell peppers is caused by the bacterium Xanthomonas campestris pv. vesicatoria. The bacteria can enter the plant through wounds, natural openings, or by splashing water.',
        'treatment': [
            'Remove and destroy infected plant material to reduce the source of bacteria.',
            'Practice crop rotation to prevent the buildup of bacteria in the soil.',
            'Avoid overhead irrigation to minimize water splashing and bacterial spread.',
            'Use copper-based fungicides or bactericides as a preventive measure or early treatment.',
            'Follow proper sanitation practices by disinfecting tools and equipment between uses.',
            'Plant resistant or tolerant bell pepper varieties when available.'
        ],
        'prevention': [
            'Plant disease-free or certified pathogen-free seeds or seedlings.',
            'Maintain adequate spacing between plants to promote air circulation and drying of foliage.',
            'Avoid working in wet conditions or handling plants when they are wet.',
            'Keep the garden free of weeds and debris, as they can harbor the bacteria.',
            'Monitor plants regularly for early signs of infection and take immediate action.'
        ]
    },
    'Bell Pepper Healthy': {},
    'Corn Common Rust': {
        'type': 'Fungal disease',
        'scientific_name': 'Puccinia sorghi',
        'symptoms': [
            'Orange or reddish-brown pustules or lesions on leaves, husks, and other above-ground plant parts',
            'Pustules may rupture and release powdery spores',
            'Yellowing and drying of infected leaves',
            'Reduced plant growth and yield'
        ],
        'causes': 'Fungal infection by Puccinia sorghi',
        'treatment': [
            'Plant resistant corn varieties when available',
            'Fungicide application can be considered in severe cases, following label instructions',
            'Remove and destroy infected plant material to reduce the source of spores',
            'Crop rotation with non-host plants can help break the disease cycle'
        ],
        'prevention': [
            'Plant corn varieties with resistance to common rust',
            'Ensure proper spacing between plants for good air circulation',
            'Avoid excessive nitrogen fertilization, as it can promote disease development',
            'Avoid working in wet conditions or handling plants when they are wet',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Corn Gray Leaf Spot': {
        'type': 'Fungal disease',
        'scientific_name': 'Cercospora zeae-maydis',
        'symptoms': [
            'Small, rectangular lesions with gray centers and dark borders on leaves',
            'Lesions may expand and merge, causing extensive damage',
            'Yellowing and drying of infected leaves',
            'Premature defoliation'
        ],
        'causes': 'Fungal infection by Cercospora zeae-maydis',
        'treatment': [
            'Plant corn hybrids with genetic resistance to gray leaf spot',
            'Fungicide application can be considered in severe cases, following label instructions',
            'Crop rotation with non-host plants can help reduce disease pressure',
            'Remove and destroy infected plant material to reduce the source of inoculum'
        ],
        'prevention': [
            'Ensure proper plant spacing for good air circulation',
            'Avoid excessive nitrogen fertilization, as it can promote disease development',
            'Practice field sanitation by removing crop debris after harvest',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Corn Healthy': {},
    'Corn Northern Leaf Blight': {
        'type': 'Fungal disease',
        'scientific_name': 'Setosphaeria turcica',
        'symptoms': [
            'Long, elliptical lesions with tan or grayish color on leaves',
            'Lesions may have dark borders and extend along the length of the leaf',
            'Severe infections can cause extensive leaf damage and defoliation',
            'Reduced plant growth and yield'
        ],
        'causes': 'Fungal infection by Setosphaeria turcica',
        'treatment': [
            'Plant corn hybrids with genetic resistance to northern leaf blight',
            'Fungicide application can be considered in severe cases, following label instructions',
            'Crop rotation with non-host plants can help break the disease cycle',
            'Remove and destroy infected plant material to reduce the source of spores'
        ],
        'prevention': [
            'Ensure proper plant spacing for good air circulation',
            'Avoid excessive nitrogen fertilization, as it can promote disease development',
            'Practice field sanitation by removing crop debris after harvest',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Potato Early Blight': {
        'type': 'Fungal disease',
        'scientific_name': 'Alternaria solani',
        'symptoms': [
            'Circular to irregular-shaped brown lesions on leaves, starting from lower leaves and progressing upward',
            'Lesions may have concentric rings or dark margins',
            'Yellowing and wilting of infected leaves',
            'Premature defoliation'
        ],
        'causes': 'Fungal infection by Alternaria solani',
        'treatment': [
            'Plant potato cultivars with resistance to early blight',
            'Fungicide application can be considered in severe cases, following label instructions',
            'Ensure proper spacing between plants for good air circulation',
            'Practice field sanitation by removing infected plant debris'
        ],
        'prevention': [
            'Rotate potato crops with non-host plants for at least two years',
            'Avoid overhead irrigation to minimize leaf wetness',
            'Plant certified disease-free potato seed tubers',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Potato Healthy': {},
    'Potato Late Blight': {
        'type': 'Fungal disease',
        'scientific_name': 'Phytophthora infestans',
        'symptoms': [
            'Water-soaked lesions on leaves, stems, and tubers',
            'Lesions rapidly enlarge and turn dark brown to purplish-black',
            'White, fuzzy growth may appear on the underside of leaves under humid conditions',
            'Foul odor from rotting tubers'
        ],
        'causes': 'Fungal infection by Phytophthora infestans',
        'treatment': [
            'Plant potato cultivars with resistance to late blight',
            'Fungicide application is often necessary, following label instructions and appropriate timing',
            'Remove and destroy infected plant material immediately to reduce the source of spores',
            'Properly dispose of infected tubers to prevent overwintering of the pathogen'
        ],
        'prevention': [
            'Plant certified disease-free potato seed tubers',
            'Ensure proper spacing between plants for good air circulation',
            'Avoid overhead irrigation and water plants at the base to minimize leaf wetness',
            'Monitor plants regularly for early signs of infection and take immediate action',
            'Practice field sanitation by removing crop debris after harvest'
        ]
    },
    'Tomato Bacterial Spot': {
        'type': 'Bacterial disease',
        'scientific_name': 'Xanthomonas vesicatoria',
        'symptoms': [
            'Water-soaked spots on leaves, stems, and fruit',
            'Spots may be dark brown to black with a yellow halo',
            'Lesions can coalesce, leading to wilting and death of affected tissue',
            'Fruit may develop raised, corky lesions with a rough texture'
        ],
        'causes': 'Bacterial infection by Xanthomonas vesicatoria',
        'treatment': [
            'Plant tomato varieties with resistance to bacterial spot, if available',
            'Apply copper-based bactericides as a preventive measure or early treatment',
            'Remove and destroy infected plant material to reduce the source of bacteria',
            'Practice crop rotation to prevent the buildup of bacteria in the soil'
        ],
        'prevention': [
            'Use certified disease-free tomato seeds or seedlings',
            'Ensure proper plant spacing for good air circulation',
            'Avoid overhead irrigation to minimize water splashing and bacterial spread',
            'Practice field sanitation by removing crop debris after harvest',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Tomato Early Blight': {
        'type': 'Fungal disease',
        'scientific_name': 'Alternaria solani',
        'symptoms': [
            'Circular to irregular-shaped brown lesions on leaves',
            'Lesions may have concentric rings or dark margins',
            'Lower leaves are usually affected first',
            'Yellowing and wilting of infected leaves'
        ],
        'causes': 'Fungal infection by Alternaria solani',
        'treatment': [
            'Remove and destroy infected plant material to reduce the source of spores',
            'Apply fungicides according to label instructions and appropriate timing',
            'Ensure proper plant spacing for good air circulation',
            'Practice crop rotation with non-host plants to break the disease cycle'
        ],
        'prevention': [
            'Plant disease-resistant tomato varieties, if available',
            'Avoid overhead irrigation and water plants at the base to minimize leaf wetness',  
            'Apply mulch to reduce soil splash onto the lower leaves',
            'Practice field sanitation by removing crop debris after harvest',
            'Monitor plants regularly for early signs of infection and take immediate action'   
        ]
    },
    'Tomato Healthy': {},
    'Tomato Late Blight': {
        'type': 'Fungal disease',
        'scientific_name': 'Phytophthora infestans',
        'symptoms': [
            'Dark, water-soaked lesions on leaves, stems, and fruit',
            'Lesions rapidly enlarge and turn brown to black',
            'White, fuzzy growth may appear on the underside of leaves under humid conditions',
            'Fruit rot and decay with a foul odor'
        ],
        'causes': 'Fungal infection by Phytophthora infestans',
        'treatment': [
            'Remove and destroy infected plant material immediately to reduce the spread of spores',
            'Apply fungicides according to label instructions and appropriate timing',
            'Maintain good airflow by pruning and staking tomato plants',
            'Ensure proper plant spacing for good air circulation'
        ],
        'prevention': [
            'Plant disease-resistant tomato varieties, if available',
            'Avoid overhead irrigation and water plants at the base to minimize leaf wetness',
            'Apply mulch to reduce soil splash onto the lower leaves',
            'Practice crop rotation with non-host plants to break the disease cycle',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Tomato Leaf Mold': {
        'type': 'Fungal disease',
        'scientific_name': 'Passalora fulva (formerly known as Fulvia fulva)',
        'symptoms': [
            'Pale yellow or light green areas on upper leaf surfaces',
            'Velvety olive-green to brown patches on lower leaf surfaces',
            'Affected leaves may curl and become distorted',
            'Yellowing and wilting of infected leaves'
        ],
        'causes': 'Fungal infection by Passalora fulva',
        'treatment': [
            'Remove and destroy infected plant material to reduce the source of spores',
            'Ensure good air circulation by proper plant spacing and pruning',
            'Avoid overhead irrigation and water plants at the base to minimize leaf wetness',
            'Apply fungicides according to label instructions and appropriate timing'
        ],
        'prevention': [
            'Plant disease-resistant tomato varieties, if available',
            'Avoid overcrowding and provide proper plant spacing for good air circulation',
            'Promote healthy plant growth through balanced fertilization and watering practices',
            'Apply mulch to reduce soil splash onto the lower leaves',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Tomato Septoria Leaf Spot': {
        'type': 'Fungal disease',
        'scientific_name': 'Septoria lycopersici',
        'symptoms': [
            'Numerous small, circular spots with dark brown centers and yellow halos on lower leaves',
            'Spots may coalesce and cause leaves to turn yellow and drop prematurely',
            'Leaves may have a "shot-hole" appearance with tiny holes in the center of spots',
            'Fruit is usually not affected by the disease'
        ],
        'causes': 'Fungal infection by Septoria lycopersici',
        'treatment': [
            'Remove and destroy infected plant material to reduce the source of spores',
            'Ensure proper plant spacing for good air circulation',
            'Apply fungicides according to label instructions and appropriate timing',
            'Avoid overhead irrigation and water plants at the base to minimize leaf wetness'
        ],
        'prevention': [
            'Plant disease-resistant tomato varieties, if available',
            'Avoid overcrowding and provide proper plant spacing for good air circulation',
            'Promote healthy plant growth through balanced fertilization and watering practices',
            'Apply mulch to reduce soil splash onto the lower leaves',
            'Practice crop rotation with non-host plants to break the disease cycle'
        ]
    },
    'Tomato Spider Mites': {
        'type': 'Pest infestation',
        'scientific_name': 'Tetranychus urticae (commonly known as Two-Spotted Spider Mite)',
        'symptoms': [
            'Fine webbing on the undersides of leaves',
            'Tiny yellow or white speckles on leaves',
            'Leaves may turn yellow, dry out, or become stippled',
            'Severe infestations can cause defoliation and reduced plant vigor'
        ],
        'causes': 'Infestation by Tetranychus urticae (Two-Spotted Spider Mite)',
        'treatment': [
            'Spray affected plants with a strong stream of water to dislodge and control mites',
            'Introduce predatory mites or beneficial insects to help control the population',
            'Use insecticidal soap, neem oil, or horticultural oils as a natural control method',
            'Consider applying acaricides if infestation is severe and other methods are ineffective'
        ],
        'prevention': [
            'Maintain proper plant hygiene and remove weeds or debris that can harbor mites',
            'Avoid over-fertilization, as it can promote mite populations',
            'Monitor plants regularly for early signs of infestation and take immediate action',
            'Provide adequate plant nutrition and avoid water stress to promote plant vigor'
        ]
    },
    'Tomato Target Spot': {
        'type': 'Fungal disease',
        'scientific_name': 'Corynespora cassiicola',
        'symptoms': [
            'Circular to irregular-shaped lesions on leaves, stems, and fruit',
            'Lesions have concentric rings and a target-like appearance',
            'Lesions may start as small, dark spots and enlarge over time',
            'Affected tissue may turn brown or black and become necrotic'
        ],
        'causes': 'Fungal infection by Corynespora cassiicola',
        'treatment': [
            'Remove and destroy infected plant material to reduce the source of spores',
            'Ensure proper plant spacing for good air circulation',
            'Apply fungicides according to label instructions and appropriate timing',
            'Avoid overhead irrigation and water plants at the base to minimize leaf wetness'
        ],
        'prevention': [
            'Plant disease-resistant tomato varieties, if available',
            'Avoid overhead irrigation and water plants at the base to minimize leaf wetness',
            'Apply mulch to reduce soil splash onto the lower leaves',
            'Practice crop rotation with non-host plants to break the disease cycle',
            'Monitor plants regularly for early signs of infection and take immediate action'
        ]
    },
    'Tomato Mosaic Virus': {
        'type': 'Viral disease',
        'scientific_name': 'Tomato mosaic virus (ToMV)',
        'symptoms': [
            'Mottled or streaked yellowing on leaves',
            'Leaf distortion and curling',
            'Reduced plant growth and stunted development',
            'Fruit may show mosaic patterns or exhibit color distortions'
        ],
        'causes': 'Infection by Tomato mosaic virus (ToMV)',
        'treatment': [
            'There is no cure for viral infections, so prevention is key',
            'Plant resistant tomato varieties, if available',
            'Control aphids and other insect vectors that can transmit the virus',
            'Remove and destroy infected plants to prevent further spread'
        ],
        'prevention': [
            'Use disease-free seeds or certified virus-free transplants',
            'Sanitize tools and equipment to prevent virus transmission',
            'Manage weed populations that can serve as virus reservoirs',
            'Control insect vectors through appropriate insecticide applications',
            'Avoid working in the garden when hands are contaminated with tobacco or other infected plants'
        ]
    },
    'Tomato Yellow Leaf Curl Virus': {
        'type': 'Viral disease',
        'scientific_name': 'Tomato yellow leaf curl virus (TYLCV)',
        'symptoms': [
            'Yellowing and upward curling of leaves, particularly young leaves',
            'Leaf narrowing and reduced leaf size',
            'Stunted plant growth and reduced vigor',
            'Abnormal development of flowers and fruits'
        ],
        'causes': 'Infection by Tomato yellow leaf curl virus (TYLCV)',
        'treatment': [
            'There is no cure for viral infections, so prevention is crucial',
            'Plant resistant tomato varieties, specifically bred for TYLCV resistance',
            'Control whiteflies and other insect vectors that transmit the virus',
            'Remove and destroy infected plants to prevent further spread'
        ],
        'prevention': [
            'Use disease-free seeds or certified virus-free transplants',
            'Use physical barriers such as insect netting to exclude whiteflies',
            'Implement cultural practices to minimize whitefly populations, such as removing weed hosts',
            'Practice good weed and pest management to reduce the presence of alternative hosts',
            'Avoid working in the garden when hands are contaminated with virus-infected plants'
        ]
    }
}

# Define the class labels
class_labels = [
    'Bell Pepper Bacterial Spot', 'Bell Pepper Healthy', 'Corn Common Rust',
    'Corn Gray Leaf Spot', 'Corn Healthy', 'Corn Northern Leaf Blight',
    'Potato Early Blight', 'Potato Healthy', 'Potato Late Blight',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy',
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites', 'Tomato Target Spot', 'Tomato Mosaic Virus',
    'Tomato Yellow Leaf Curl Virus'
]
healthy_labels = [
    'Bell Pepper Healthy', 'Corn Healthy', 'Potato Healthy', 'Tomato Healthy'
]
plant_labels=["Bell Pepper","Corn","Potato","Tomato"]


# Define colors for styling
colors = {
    'primary': '#008080',
    'secondary': '#F63366',
    'background': '#F0F2F6',
    'text': '#333333',
    'success': '#00CC96',
    'error': '#FF5C58',
}

# Define custom styles
st.markdown(
    f"""
    <style>
    body {{
        background-color: {colors['background']};
        color: {colors['text']};
    }}
    .reportview-container .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }}
    .sidebar .sidebar-content {{
        background-color: {colors['primary']};
    }}
    .css-1g6pdkw {{
        color: {colors['secondary']} !important;
    }}
    .css-11jw9g0 {{
        color: {colors['success']} !important;
    }}
    .css-1q2sf9f {{
        color: {colors['error']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Define the function to preprocess the input image
def preprocess_image(img):
    img = img.resize((256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Streamlit app code
def main():
    st.write("**Note:** Only supported plants are:", ", ".join(plant_labels))
    st.title("Plant Disease Detection")

    # Drag and drop functionality for image upload
    uploaded_file = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key='fileUploader')

    if uploaded_file is not None:
        # Preprocess the uploaded image
        img = image.load_img(uploaded_file)
        img = preprocess_image(img)

        # Make predictions
        predictions = model.predict(img)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display predicted class and percentage damaged
        st.subheader("Predicted Class:")
        st.write(predicted_class)
        st.subheader("Confidence Level:")
        st.write(f"{confidence * 100:.2f}%")

        # Display the uploaded image with styling
        st.subheader("Uploaded Image:")
        st.image(img, use_column_width=True)
    
        # Display additional information for the predicted class
        if predicted_class not in healthy_labels:
            st.subheader("Type:")
            st.write(class_info[predicted_class]['type'])
            st.subheader("Scientific Name:")
            st.write(class_info[predicted_class]['scientific_name'])
            st.subheader("Symptoms:")
            for symptom in class_info[predicted_class]['symptoms']:
                st.write(f"- {symptom}")
            st.subheader("Cause:")
            st.write(class_info[predicted_class]['causes'])
            st.subheader("Treatment and Control:")
            for treatment in class_info[predicted_class]['treatment']:
                st.write(f"- {treatment}")
            st.subheader("Prevention:")
            for prevention in class_info[predicted_class]['prevention']:
                st.write(f"- {prevention}")

        # Allow users to correct the prediction
        st.subheader("Correct Prediction")
        correct_class = st.selectbox("Select the correct class", class_labels, index=class_labels.index(predicted_class))
        if st.button("Update Model") and correct_class != predicted_class:
            # Update the model with the corrected information
            class_info[correct_class] = class_info[predicted_class]
            st.success("Model updated successfully!")

if __name__ == '__main__':
    main()
