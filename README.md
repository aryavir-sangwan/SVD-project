# SVD-project
The goal of this project is to reconstruct a grayscale image using singular value decomposition. I will elaborate on the math first, and then analyze the results. 
Let A be a matrix in R^(m x n) representing our test image. Each entry represents the color grade of a pixel in the image. 255 corresponds to white, and 0 corresponds to black. Values in between are different shades of gray. Our goal is to reconstruct our test image as closely as possible using a low rank approximation of A. 
First let us derive the SVD formula, which is $A= (U)(\Sigma)(V^T)$ 
