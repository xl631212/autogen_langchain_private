
# NexaAgent 0.0.1

![NexaAgent Preview](1.png)

## app links: 
### https://huggingface.co/spaces/xuyingliKepler/nexaagent

## Introduction

NexaAgent 0.0.1 is a state-of-the-art tool designed for processing and interacting with PDF documents using advanced technologies like AutoGen, LangChain, and chromadb. Whether you're looking to extract information, answer questions, or perform any other task related to a PDF, NexaAgent is here to assist.


## Features

- **Upload and Process PDFs**: Seamlessly upload any PDF document and let NexaAgent handle the rest.
- **Interactive Question-Answer System**: Pose questions related to the uploaded PDF and receive precise answers.
- **Advanced Output Capture**: Efficiently captures and logs interactions for future reference.
- **Extended User Proxy**: Enhanced user-agent interactions with logging capabilities.
- **Powered by Advanced Technologies**: Utilizes the power of AutoGen, LangChain, chromadb, and more to ensure top-notch results.

## How to Use

1. **Start the Application**: Execute the script to initiate the NexaAgent interface.
2. **Upload a PDF**: Utilize the file uploader to select and upload a desired PDF document.
3. **Input Your Task**: In the designated text area, input the task or question related to the uploaded PDF.
4. **Retrieve the Answer**: NexaAgent will process the PDF and present you with the answer or result in the adjacent text area.

## Setup and Running

### Prerequisites

Before running NexaAgent, ensure you have the following prerequisites installed:

- Python 3.x
- Streamlit
- AutoGen
- OpenAI

### Installation

1. **Clone the Repository**:
   ```bash
   git clone ...
   ```

2. **Install Required Libraries**:
   ```bash
   pip install streamlit autogen openai
   ```

3. **Set Up OpenAI API Key**:
   Ensure you have your OpenAI API key set up. You can either set it as an environment variable or use Streamlit's secrets management.

4. **Run the Application**:
   ```bash
   streamlit run <filename>.py
   ```

### Configuration

- **OpenAI API Key**: The application requires an OpenAI API key to function. Ensure you have it set up in `st.secrets["OPENAI_API_KEY"]`.

- **PDF Processing**: The application is designed to process PDFs. Ensure the PDFs you're using are compatible and are not password protected for optimal results.

### Troubleshooting

- **PDF Upload Issues**: If you encounter issues while uploading a PDF, ensure the file is not corrupted and try again.
  
- **API Limitations**: If you're using the free tier of the OpenAI API, be aware of the limitations in terms of the number of requests and response time.

## Contributing

We welcome contributions from the community. If you'd like to improve the application, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. For more details, refer to the `LICENSE` file in the repository.

## Acknowledgements

Special thanks to the developers of Streamlit, AutoGen, and OpenAI for their incredible tools that made this project possible.
