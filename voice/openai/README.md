# @mastra/voice-openai

OpenAI Voice integration for Mastra, providing both Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities using OpenAI's advanced models.

## Installation

```bash
npm install @mastra/voice-openai
```

## Configuration

The module requires an OpenAI API key, which can be provided through environment variables or directly in the configuration:

```bash
OPENAI_API_KEY=your_api_key
```

## Usage

### Using the Factory Function (Recommended)

```typescript
import { createOpenAIVoice } from '@mastra/voice-openai';

// Create voice with both speech and listening capabilities
const voice = createOpenAIVoice({
  speech: {
    model: 'tts-1', // or 'tts-1-hd' for higher quality
    apiKey: 'your-api-key', // Optional, can use OPENAI_API_KEY env var
    speaker: 'alloy', // Default voice
  },
  listening: {
    model: 'whisper-1',
    apiKey: 'your-api-key', // Optional, can use OPENAI_API_KEY env var
  },
});

// Or create speech-only voice
const speechVoice = createOpenAIVoice({
  speech: {
    model: 'tts-1',
    speaker: 'nova',
  },
});

// Or create listening-only voice
const listeningVoice = createOpenAIVoice({
  listening: {
    model: 'whisper-1',
  },
});

// List available voices
const speakers = await voice.getSpeakers();

// Generate speech
const audioStream = await voice.speak('Hello from Mastra!');

// Convert speech to text
const text = await voice.listen(audioStream, {
  filetype: 'wav',
});
```

### Using the Class Directly

```typescript
import { OpenAIVoice } from '@mastra/voice-openai';

const voice = new OpenAIVoice({
  speechModel: {
    name: 'tts-1',
    apiKey: 'your-api-key', // Optional, can use OPENAI_API_KEY env var
  },
  listeningModel: {
    name: 'whisper-1',
    apiKey: 'your-api-key', // Optional, can use OPENAI_API_KEY env var
  },
  speaker: 'alloy', // Default voice
});
```

## Features

- High-quality Text-to-Speech synthesis
- Accurate Speech-to-Text transcription
- Multiple voice options
- Natural and expressive speech output
- Fast processing times

## Voice Options

OpenAI provides several high-quality voices:

- alloy (Neutral)
- echo (Male)
- fable (Male)
- onyx (Male)
- nova (Female)
- shimmer (Female)
- ash (Male)
- coral (Female)
- sage (Male)

View the complete list in OpenAI's [Text to Speech documentation](https://platform.openai.com/docs/guides/text-to-speech).
