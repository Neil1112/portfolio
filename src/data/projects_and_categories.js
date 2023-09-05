import NeuralNetworksImg from '../assets/neural-networks.jpg';

const projects = [
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
    name: 'Generative Adversarial Networks (GANs)',
    projects: [
    projects[0],
    ],
},
];


export { projects, project_categories };