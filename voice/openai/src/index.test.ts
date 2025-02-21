import { writeFileSync, mkdirSync, createReadStream } from 'fs';
import path from 'path';
import { PassThrough } from 'stream';
import { describe, expect, it, beforeAll, beforeEach } from 'vitest';

import { createOpenAIVoice, OpenAIVoice } from './index.js';

describe('createOpenAIVoice', () => {
  const outputDir = path.join(process.cwd(), 'test-outputs');
  let apiKey: string;

  beforeAll(() => {
    try {
      mkdirSync(outputDir, { recursive: true });
    } catch (err) {
      // Ignore if directory already exists
    }
  });

  beforeEach(() => {
    apiKey = process.env.OPENAI_API_KEY || '';
  });

  it('should create voice with speech capabilities and generate audio', async () => {
    const voice = createOpenAIVoice({
      speech: {
        model: 'tts-1',
        speaker: 'alloy',
      },
    });

    expect(voice.speak).toBeDefined();
    expect(voice.getSpeakers).toBeDefined();
    expect(voice.listen).toBeUndefined();

    const audioStream = await voice.speak!('Testing speech capabilities');
    const chunks: Buffer[] = [];
    for await (const chunk of audioStream) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    const audioBuffer = Buffer.concat(chunks);

    expect(audioBuffer.length).toBeGreaterThan(0);
    writeFileSync(path.join(outputDir, 'factory-speech-test.mp3'), audioBuffer);
  }, 10000);

  it('should create voice with listening capabilities and transcribe audio', async () => {
    const speechVoice = createOpenAIVoice({
      speech: { model: 'tts-1' },
    });

    const audioStream = await speechVoice.speak!('Testing transcription capabilities');
    const voice = createOpenAIVoice({
      listening: {
        model: 'whisper-1',
      },
    });

    expect(voice.listen).toBeDefined();
    expect(voice.speak).toBeUndefined();

    const text = await voice.listen!(audioStream, { filetype: 'mp3' });
    expect(text.toLowerCase()).toContain('testing');
    expect(text.toLowerCase()).toContain('transcription');
  }, 15000);

  it('should create voice with both capabilities and round-trip audio', async () => {
    const voice = createOpenAIVoice({
      speech: {
        model: 'tts-1',
        speaker: 'alloy',
      },
      listening: {
        model: 'whisper-1',
      },
    });

    expect(voice.speak).toBeDefined();
    expect(voice.listen).toBeDefined();

    const originalText = 'Testing both speech and listening capabilities';
    const audioStream = await voice.speak!(originalText);
    const transcribedText = await voice.listen!(audioStream, { filetype: 'mp3' });

    expect(transcribedText.toLowerCase()).toContain('testing');
    expect(transcribedText.toLowerCase()).toContain('capabilities');
  }, 20000);

  it('should list available speakers', async () => {
    const voice = createOpenAIVoice({
      speech: {
        model: 'tts-1',
      },
    });

    const speakers = await voice.getSpeakers!();
    expect(speakers).toContainEqual({ voiceId: 'alloy' });
    expect(speakers).toContainEqual({ voiceId: 'nova' });
    expect(speakers.length).toBeGreaterThan(0);
  });

  it('should create voice without any capabilities', () => {
    const voice = createOpenAIVoice();

    expect(voice.speak).toBeUndefined();
    expect(voice.listen).toBeUndefined();
    expect(voice.getSpeakers).toBeUndefined();
  });
});

describe('OpenAIVoice Integration Tests', () => {
  let voice: OpenAIVoice;
  const outputDir = path.join(process.cwd(), 'test-outputs');

  beforeAll(() => {
    // Create output directory if it doesn't exist
    try {
      mkdirSync(outputDir, { recursive: true });
    } catch (err) {
      // Ignore if directory already exists
    }

    voice = new OpenAIVoice({
      speechModel: {
        name: 'tts-1',
      },
      listeningModel: {
        name: 'whisper-1',
      },
    });
  });

  describe('getSpeakers', () => {
    it('should list available voices', async () => {
      const speakers = await voice.getSpeakers();
      expect(speakers).toContainEqual({ voiceId: 'alloy' });
      expect(speakers).toContainEqual({ voiceId: 'nova' });
    });
  });

  describe('speak', () => {
    it('should generate audio stream from text', async () => {
      const audioStream = await voice.speak('Hello World', {
        speaker: 'alloy',
      });

      const chunks: Buffer[] = [];
      for await (const chunk of audioStream) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      const audioBuffer = Buffer.concat(chunks);

      expect(audioBuffer.length).toBeGreaterThan(0);

      const outputPath = path.join(outputDir, 'speech-test.mp3');
      writeFileSync(outputPath, audioBuffer);
    }, 10000);

    it('should work with different parameters', async () => {
      const audioStream = await voice.speak('Test with parameters', {
        speaker: 'nova',
        speed: 0.5,
      });

      const chunks: Buffer[] = [];
      for await (const chunk of audioStream) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      const audioBuffer = Buffer.concat(chunks);

      expect(audioBuffer.length).toBeGreaterThan(0);

      const outputPath = path.join(outputDir, 'speech-test-params.mp3');
      writeFileSync(outputPath, audioBuffer);
    }, 10000);

    it('should accept text stream as input', async () => {
      const inputStream = new PassThrough();
      inputStream.end('Hello from stream');

      const audioStream = await voice.speak(inputStream, {
        speaker: 'alloy',
      });

      const chunks: Buffer[] = [];
      for await (const chunk of audioStream) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      const audioBuffer = Buffer.concat(chunks);

      expect(audioBuffer.length).toBeGreaterThan(0);

      const outputPath = path.join(outputDir, 'speech-stream-input.mp3');
      writeFileSync(outputPath, audioBuffer);
    }, 10000);
  });

  describe('listen', () => {
    it('should transcribe audio from fixture file', async () => {
      const fixturePath = path.join(process.cwd(), '__fixtures__', 'voice-test.m4a');
      const audioStream = createReadStream(fixturePath);

      const text = await voice.listen(audioStream, {
        filetype: 'm4a',
      });

      expect(text).toBeTruthy();
      console.log(text);
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
    }, 15000);

    it('should transcribe audio stream', async () => {
      // First generate some test audio
      const audioStream = await voice.speak('This is a test for transcription', {
        speaker: 'alloy',
      });

      // Then transcribe it
      const text = await voice.listen(audioStream, {
        filetype: 'm4a',
      });

      expect(text).toBeTruthy();
      expect(typeof text).toBe('string');
      expect(text.toLowerCase()).toContain('test');
    }, 15000);

    it('should accept options', async () => {
      const audioStream = await voice.speak('Test with language option', {
        speaker: 'nova',
      });

      const text = await voice.listen(audioStream, {
        language: 'en',
        filetype: 'm4a',
      });

      expect(text).toBeTruthy();
      expect(typeof text).toBe('string');
      expect(text.toLowerCase()).toContain('test');
    }, 15000);
  });

  // Error cases
  describe('error handling', () => {
    it('should handle invalid speaker names', async () => {
      await expect(
        voice.speak('Test', {
          speaker: 'invalid_voice',
        }),
      ).rejects.toThrow();
    });

    it('should handle empty text', async () => {
      await expect(
        voice.speak('', {
          speaker: 'alloy',
        }),
      ).rejects.toThrow();
    });
  });
});
