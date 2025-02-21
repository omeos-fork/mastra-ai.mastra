import { MastraVoice } from '@mastra/core/voice';
import OpenAI from 'openai';
import { PassThrough } from 'stream';

type OpenAIVoiceId = 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer' | 'ash' | 'coral' | 'sage';
type OpenAIModel = 'tts-1' | 'tts-1-hd' | 'whisper-1';
type OpenAIFileType = 'mp3' | 'mp4' | 'mpeg' | 'mpga' | 'm4a' | 'wav' | 'webm';

export interface OpenAIConfig {
  name?: OpenAIModel;
  apiKey?: string;
}

export interface OpenAIVoiceConfig {
  speech?: {
    model: 'tts-1' | 'tts-1-hd';
    apiKey?: string;
    speaker?: OpenAIVoiceId;
  };
  listening?: {
    model: 'whisper-1';
    apiKey?: string;
    language?: string;
  };
}

export interface OpenAIVoiceCapabilities {
  speak?: (
    input: string | NodeJS.ReadableStream,
    options?: { speaker?: string; speed?: number },
  ) => Promise<NodeJS.ReadableStream>;
  listen?: (
    audioStream: NodeJS.ReadableStream,
    options: { filetype: OpenAIFileType; [key: string]: any },
  ) => Promise<string>;
  getSpeakers?: () => Promise<Array<{ voiceId: OpenAIVoiceId }>>;
}

/**
 * Creates OpenAI voice capabilities
 */
export function createOpenAIVoice(config?: OpenAIVoiceConfig): OpenAIVoiceCapabilities {
  const provider = new OpenAIVoice({
    speechModel: config?.speech
      ? {
          name: config.speech.model,
          apiKey: config.speech.apiKey,
        }
      : undefined,
    listeningModel: config?.listening
      ? {
          name: config.listening.model,
          apiKey: config.listening.apiKey,
        }
      : undefined,
    speaker: config?.speech?.speaker,
  });

  return {
    speak: config?.speech ? provider.speak.bind(provider) : undefined,
    listen: config?.listening ? provider.listen.bind(provider) : undefined,
    getSpeakers: config?.speech ? provider.getSpeakers.bind(provider) : undefined,
  };
}

export class OpenAIVoice extends MastraVoice {
  speechClient?: OpenAI;
  listeningClient?: OpenAI;

  constructor({
    listeningModel,
    speechModel,
    speaker,
  }: {
    listeningModel?: OpenAIConfig;
    speechModel?: OpenAIConfig;
    speaker?: string;
  }) {
    super({
      speechModel: speechModel && {
        name: speechModel.name || 'tts-1',
        apiKey: speechModel.apiKey,
      },
      listeningModel: listeningModel && {
        name: listeningModel.name || 'whisper-1',
        apiKey: listeningModel.apiKey,
      },
      speaker,
    });

    const defaultApiKey = process.env.OPENAI_API_KEY;

    if (speechModel || defaultApiKey) {
      const speechApiKey = speechModel?.apiKey || defaultApiKey;
      if (!speechApiKey) {
        throw new Error('No API key provided for speech model');
      }
      this.speechClient = new OpenAI({ apiKey: speechApiKey });
    }

    if (listeningModel || defaultApiKey) {
      const listeningApiKey = listeningModel?.apiKey || defaultApiKey;
      if (!listeningApiKey) {
        throw new Error('No API key provided for listening model');
      }
      this.listeningClient = new OpenAI({ apiKey: listeningApiKey });
    }

    if (!this.speechClient && !this.listeningClient) {
      throw new Error('At least one of OPENAI_API_KEY, speechModel.apiKey, or listeningModel.apiKey must be set');
    }
  }

  async getSpeakers(): Promise<Array<{ voiceId: OpenAIVoiceId }>> {
    if (!this.speechModel) {
      throw new Error('Speech model not configured');
    }

    return [
      { voiceId: 'alloy' },
      { voiceId: 'echo' },
      { voiceId: 'fable' },
      { voiceId: 'onyx' },
      { voiceId: 'nova' },
      { voiceId: 'shimmer' },
      { voiceId: 'ash' },
      { voiceId: 'coral' },
      { voiceId: 'sage' },
    ];
  }

  async speak(
    input: string | NodeJS.ReadableStream,
    options?: {
      speaker?: string;
      speed?: number;
      [key: string]: any;
    },
  ): Promise<NodeJS.ReadableStream> {
    if (!this.speechClient) {
      throw new Error('Speech model not configured');
    }

    if (typeof input !== 'string') {
      const chunks: Buffer[] = [];
      for await (const chunk of input) {
        chunks.push(Buffer.from(chunk));
      }
      input = Buffer.concat(chunks).toString('utf-8');
    }

    if (input.trim().length === 0) {
      throw new Error('Input text is empty');
    }

    const audio = await this.traced(async () => {
      const response = await this.speechClient!.audio.speech.create({
        model: this.speechModel?.name || 'tts-1',
        voice: (options?.speaker || 'alloy') as OpenAIVoiceId,
        input,
        speed: options?.speed || 1.0,
      });

      const passThrough = new PassThrough();
      const buffer = Buffer.from(await response.arrayBuffer());
      passThrough.end(buffer);
      return passThrough;
    }, 'voice.openai.speak')();

    return audio;
  }

  async listen(
    audioStream: NodeJS.ReadableStream,
    options: {
      filetype: 'mp3' | 'mp4' | 'mpeg' | 'mpga' | 'm4a' | 'wav' | 'webm';
      [key: string]: any;
    },
  ): Promise<string> {
    if (!this.listeningClient) {
      throw new Error('Listening model not configured');
    }

    const chunks: Buffer[] = [];
    for await (const chunk of audioStream) {
      chunks.push(Buffer.from(chunk));
    }
    const audioBuffer = Buffer.concat(chunks);

    const text = await this.traced(async () => {
      const { filetype, ...otherOptions } = options || {};
      const file = new File([audioBuffer], `audio.${filetype}`);

      const response = await this.listeningClient!.audio.transcriptions.create({
        model: this.listeningModel?.name || 'whisper-1',
        file: file as any,
        ...otherOptions,
      });

      return response.text;
    }, 'voice.openai.listen')();

    return text;
  }
}
