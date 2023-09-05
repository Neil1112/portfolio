export const markdownContent = `
# Week 1 notes compiled

---

# Generative Models

## Generative vs Discriminative Models

**Discriminative Models -**

- Classifiers - Dog or Cat
- $x \rightarrow y$
    - $x: features$,
    - $y: class$
- They give $P(y|x)$

**Generative Models -**

- They learn how to make a realistic representation of some class.
    - e.g. - Image of a dog
- They take some random noise input $\{xi}$ and sometimes a class $y$ and generates a set of features $x$.
- $\{xi}, y \rightarrow x$
    - $\{xi}: noise$,
    - $x: features$,
    - $y: class$
- They give $P(x|y)$
    - $y$ can be omitted if there is only 1 class.

There are many types of Generative models - 

## Variational Autoencoders (VAEs)

- Encoder’s job is to transform the image as a vector in the latent space.
- Then we sample a nearby vector to this vector and put it through the Decoder to reconstruct the realistic image that the encoder saw before.
- After training, we only use the Decoder. We take a random sample and construct the image through the Decoder.
- The VAE injects some noise as well.
    - So instead of putting the point in latent space, the encoder actually encodes the image onto a whole distribution and then samples a point on that distribution to feed into the decoder.

## Generative Adversarial Networks (GANs)

- They work a bit different from VAEs.
- Generator generates the images like the decoder by taking in some noise.
    - It can also take optionally a class (if there is only 1 class, then we can omit inputting it).
- The Discriminator looks at the fake and real images simultaneously and figure out which are real and fakes.
- These models competes against each other. Hence the name “Adversarial”.
- After training, we won’t need the Discriminator Model anymore.

## Summary

- Generative Models learn to produce realistic examples.
- Discriminative models distinguish between classes (or real / fakes).

---

# Intuition Behind GANs

- GANs have 2 components :
    - Generator
    - Discriminator
- These are typically Neural Networks.
- Generator learns to make fakes that looks real.
- Discriminator learns to distinguish real from fake.
- The learn from the competition with each other.
- At the end, fakes look real.

---

# Discriminator

- Type of classifier.
- The Goal of the discriminator is to model the probability of each class given a set of input features.
    - $P(cat|image)$
    - i.e.- $P(y|x)$
- The Discriminator takes in the “fake” image and classify how fake it is.
    - e.g.- $P(fake|image) = 0.85$
- i.e.-  for Discriminator : $P(fake|image)$

## Summary

- The Discriminator is a classifier.
- It learns the probability of class $y$ (real or fake) given features $x$.
- The probabilities are the feedback for the Generator.

---
`