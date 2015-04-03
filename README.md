CUDA/C++ implementation of several papers in the spirit of developing a small demo similar to Nvidia's FleX framework (https://developer.nvidia.com/physx-flex)

Papers implemented in part or in full:
  - Position Based Fluids - http://mmacklin.com/pbf_sig_preprint.pdf
  - Position Based Dynamics - http://matthias-mueller-fischer.ch/publications/posBasedDyn.pdf
  - Unified Particle Physics - http://mmacklin.com/uppfrta_preprint.pdf
  - Long Range Attachments - http://matthias-mueller-fischer.ch/publications/sca2012cloth.pdf
  - Unified Spray, Foam and Bubbles for Particle-Based Fluids - http://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf
  - Screen Space Foam Rendering - http://cg.informatik.uni-freiburg.de/publications/2013_WSCG_foamRendering.pdf
  - Screen Space Fluid Rendering - http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf
  
On current hardware a fluid scene of 50k fluid particles and up to 500k diffuse particles runs at about 10-15 fps. The cloth scene which consists of a cloth mesh of 4k vertices, approximately 1k fluid particles, and up to 20k diffuse particles runs at 25-30 fps. Additional parameter tuning and optimizations could be done to make it even faster.

Dependencies for the project include:
  - GLEW
  - GLFW
  - CUDA 7
  - GLM
  - DevIL


Screenshots:
![Alt text](/PositionBasedFluids/fluid.png?raw=true "Fluid Scene")
![Alt text](/PositionBasedFluids/cloth.png?raw=true "Cloth Scene")
