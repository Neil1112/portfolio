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
    name: 'Miscellaneous',
    projects: [
      projects[0],
      projects[1],
      projects[2],
    ],
  },

  {
      name: 'Generative Adversarial Networks (GANs)',
      projects: [
      projects[3],
      ],
  },
];


export { projects, project_categories };