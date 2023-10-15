import NeuralNetworksImg from '../assets/neural-networks.jpg';

const projects = [
    {
      id: 'flan-t5-fine-tuning',
      title: 'Flan T5 Fine Tuning for Dialogue Summarisation',
      description: 'I fine tuned FLAN-T5 from Hugging Face for enhanced dialogue summarisation. The FLAN-T5 provides a high quality instruction tuned model and can summarise text out of the box. To improve the inferences, I applied Parameter Efficient Fine-Tuning (PEFT), particularly the Low Rank Adapter (LoRA) technique. I performed both quantitative (human) and qualitative evaluation using ROUGE metrics.',
      image: NeuralNetworksImg,
    },

    {
      id: 'neural-style-transfer',
      title: 'Neural Style Transfer',
      description: 'Neural Style Transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.',
      image: NeuralNetworksImg,
    },

    {
      id: 'recommender-system-collaborative-filtering',
      title: 'Recommender System - Collaborative Filtering',
      description: 'I applied the Collaborative Filtering learning algorithm to the MovieLens Dataset. The goal of a collaborative filtering recommender system is to generate two vectors. For each user a parameter vector that embodies the movie taste of a user. For each movie a feature vector of the same size which embodies some description of the movie. The dot product of the two vectors plus the bias term should produce an estimate of the rating the user might give to that movie.',
      image: NeuralNetworksImg,
    },

    // {
    //   id: 'human-image-segmentation-unet-resnet18',
    //   title: 'Human Image Segmentation - U-Net ResNet18',
    //   description: 'I applied the U-Net architecture with ResNet18 as Encoder to detect Human Segmentation Masks using PyTorch. The Dataset contained both Real Images with their Labelled Masks. I first applied various augmentations to the dataset and then used a combination Dice Loss and Binary Cross Entropy Loss as the loss function. The ResNet18 Encoder used weights from the ImageNet Dataset.',
    //   image: NeuralNetworksImg,
    // },

    {
      id: 'svd-image-compression',
      title: 'SVD Image Compression',
      description: 'I applied Singular Value Decomposition (SVD) to compress images. SVD is a matrix factorization technique which can be used to reduce the dimensionality of the data. It is also used for image compression.',
      image: NeuralNetworksImg,
    },

    {
      id: 'tic-tac-toe',
      title: 'Tic Tac Toe - Minimax Algorithm',
      description: 'I implemented the Minimax Algorithm to create an unbeatable Tic Tac Toe AI. The Minimax Algorithm is a recursive algorithm which is used to choose an optimal move for a player assuming that the opponent is also playing optimally. It is used in two player games such as Tic Tac Toe, Chess, etc.',
      image: NeuralNetworksImg,
    },

    {
      id: 'gpt-knowledge-extension',
      title: 'GPT Knowledge Extension',
      description: 'I extended the knowledge of GPT-3.5 using Retrieval Augmented Generation (RAG) method using Langchain and ChromaDB as Vector database.',
      image: NeuralNetworksImg,
    },

    // Generative AI
    {
      id: 'gan-implementation-in-pytorch',
      title: 'GAN Implementation in PyTorch',
      description: 'Generative Adversarial Networks (GANs) are widely used to generate realistic images. They use two neural networks which competes against each other. One is Generator and other is Discriminator.',
      image: NeuralNetworksImg,
    },
    
      
];
  
const project_categories = [
  {
    name: 'Large Language Models',
    projects: [
      projects[0],
      projects[5],
    ],
  },

  {
      name: 'Generative Adversarial Networks (GANs)',
      projects: [
      projects[6],
      ],
  },

  {
    name: 'Miscellaneous',
    projects: [
      projects[3],
      projects[2],
      projects[1],
      projects[4],
      projects[5],

    ],
  },
];


export { projects, project_categories };