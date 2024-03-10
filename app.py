import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import streamlit as st

# Title and User Input
st.title("Text-to-Video with Streamlit")
prompt = st.text_input("Enter your text prompt:", "Spiderman is surfing")

# Button to trigger generation
if st.button("Generate Video"):

    # Ensure you have 'accelerate' version 0.17.0 or higher (see previous explanation)
    import accelerate
    if accelerate.__version__ < "0.17.0":
        st.warning("Please upgrade 'accelerate' to version 0.17.0 or higher for CPU offloading.")
    else:
        with st.spinner("Generating video..."):
            pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", 
                                         torch_dtype=torch.float16, 
                                         variant="fp16",
                                         device="cpu") # Force CPU usage
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()  # Assuming 'accelerate' is updated  

            video_frames = pipe(prompt, num_inference_steps=25).frames
            video_path = export_to_video(video_frames)

            # Display the video in the Streamlit app
            st.video(video_path)
