# 🎼 Music Transcription Service 🎙️

Dive into the world of music transcription! Our tool is designed to transcribe music and compare singing performances with lyrics, all powered by gRPC.

---

## 🌟 **Features**

- **Versatility**: Choose from multiple transcription engines - OpenAI's Whisper, Google Speech, and Web2Vec2.
- **Precision**: Tailored specifically for music, ensuring accurate transcriptions.
- **Ease of Use**: Simple setup and execution with Docker.

---

## 🚀 **Quick Start**

1. **Setup the gRPC Image**
   ```bash
   rm -r ./proto/Generated
   docker build -t grpc-builder -f Dockerfile.grpcgen .
   ```

2. **Initialize a Temporary Container**
   ```bash
   docker run --name grpc-temp grpc-builder /bin/true
   ```

3. **Retrieve the Generated Files**
   ```bash
   docker cp grpc-temp:/app/Generated ./proto
   ```

4. **Cleanup: Remove Temporary Container**
   ```bash
   docker rm grpc-temp
   ```

5. **Update the gRPC Files (if needed)**
   ```python
   from generated import audio_transcription_pb2 as audio__transcription__pb2
   ```

6. **Launch with Docker Compose**
   ```bash
   docker-compose build
   docker-compose up
   ```
   For client operations:
   ```bash
   docker-compose build grpc-client
   docker-compose up grpc-client
   ```