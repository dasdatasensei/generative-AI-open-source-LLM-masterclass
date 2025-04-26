# The Complete GenAI & RAG Masterclass
## Build Production-Ready AI Systems with Python | LangChain, Vector Databases & LLM Integration

A comprehensive, project-based course teaching how to build practical GenAI applications using free, open-source LLMs, RAG systems with multiple vector databases, and LangChain integration.

## Course Overview

This course emphasizes hands-on implementation with real-world projects, using multiple open-source models to demonstrate different capabilities and resource requirements.

### What You'll Learn

- Build practical applications with open-source LLMs (Llama 3, Mistral, Phi-3)
- Implement advanced RAG systems with multiple vector database technologies
- Integrate LangChain for document processing and agent building
- Create production-ready deployments with monitoring and scaling
- Optimize models for consumer hardware through quantization
- Implement enterprise-grade security and performance practices
- Fine-tune models for specialized business needs
- Evaluate and troubleshoot GenAI applications

### Course Structure

1. **Introduction to GenAI with Open-Source LLMs**
   - Setup and environment configuration
   - Understanding the open-source LLM ecosystem
   - First project: Simple Q&A System with Phi-3

2. **Understanding Open-Source LLM Basics**
   - Model quantization techniques
   - Hardware requirements and optimization
   - Project: Build a specialized coding assistant with Mistral 7B

3. **RAG Fundamentals and Vector Databases**
   - Vector databases compared: FAISS, Chroma, Weaviate
   - Document processing and chunking strategies
   - Project: Document QA system with Llama 3 8B and Chroma DB

4. **Advanced RAG Techniques and Optimization**
   - Hybrid search and retrieval optimization
   - Re-ranking and context compression
   - Project: Multi-source research assistant with Mixtral 8x7B

5. **LangChain Integration for RAG and LLMs**
   - Document loaders and text splitters
   - Custom chain and tool development
   - Project: Multi-step reasoning agent with LangChain and WizardLM

6. **Building a Complete GenAI Web Application**
   - Full-stack implementation with Gradio and FastAPI
   - Real-time monitoring and logging integration
   - Project: Multi-model GenAI chat application with monitoring

7. **Production Deployment and Scaling**
   - Containerization and optimization
   - Scaling strategies and monitoring infrastructure
   - Project: Deploy your application with monitoring and auto-scaling

8. **Fine-tuning and Customizing Open-Source LLMs**
   - LoRA and QLoRA techniques
   - Training data preparation
   - Project: Deploy your own fine-tuned model

9. **Advanced Integration Patterns and Enterprise Deployment**
   - Authentication and access control
   - Multi-tenant deployment considerations
   - Project: Enterprise-ready API wrapper with usage analytics

10. **Comprehensive Evaluation of LLMs and RAG Systems**
    - Evaluation frameworks and methodologies
    - Factuality and hallucination detection
    - Project: Build an evaluation dashboard with automated reporting

11. **Troubleshooting GenAI Applications - Common Errors and Solutions**
    - Debugging RAG hallucinations and retrieval problems
    - Performance optimization for slow inference
    - Project: Create a comprehensive troubleshooting guide

## Getting Started

### Hardware Requirements

- **Minimal:** CPU-only with 16GB RAM (will run Phi-3 Mini and small Mistral models)
- **Recommended:** CUDA-compatible GPU with 8GB VRAM
- **Optimal:** CUDA-compatible GPU with 16GB+ VRAM (for Llama 3 70B quantized)

### Setup Instructions

#### Local Development (VSCode)

1. Clone this repository:
```bash
git clone https://github.com/dasdatasensei/generative-AI-open-source-LLM-masterclass
cd generative-AI-open-source-LLM-masterclass
```

2. Run the setup script:
```bash
python setup.py  # Basic setup
# OR
python setup.py --full  # Complete setup with all dependencies
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

#### GPU-Intensive Tasks (Google Colab)

For larger models and GPU-intensive tasks, we provide Google Colab notebooks. These can be accessed directly from the GitHub repository through Colab's "Open in Colab" feature.

## Repository Structure

- Each lesson has its own directory with slides, notebooks, and project materials
- Both VSCode and Colab versions of notebooks are provided
- Common utilities and helper functions are in the `utils` directory
- Model configuration files are in the `configs` directory

## Contact

- GitHub: [github.com/dasdatasensei](https://github.com/dasdatasensei)
