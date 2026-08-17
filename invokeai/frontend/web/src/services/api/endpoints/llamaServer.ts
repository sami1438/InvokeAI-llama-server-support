import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the llama-server router
 * @example
 * buildLlamaServerUrl('test')
 * // '/api/v1/llm/external/test'
 */
const buildLlamaServerUrl = (path: string = '') => buildV1Url(`llm/external/${path}`);

// --- Types ---

export type LlamaServerConfig = {
  enabled: boolean;
  url: string | null;
  text_model: string | null;
  vision_model: string | null;
  timeout: number;
  max_tokens: number;
  temperature: number;
  api_key_configured: boolean;
};

export type LlamaServerConfigUpdate = {
  enabled?: boolean;
  url?: string | null;
  api_key?: string | null;
  text_model?: string | null;
  vision_model?: string | null;
  timeout?: number;
  max_tokens?: number;
  temperature?: number;
};

export type LlamaServerTestResult = {
  ok: boolean;
  models?: string[];
  error?: string;
};

export type LlamaServerExpandPromptRequest = {
  prompt: string;
  system_prompt?: string | null;
  model?: string | null;
  max_tokens?: number;
  temperature?: number;
  task_id?: string | null;
};

export type LlamaServerExpandPromptResponse = {
  expanded_prompt: string;
  error?: string | null;
};

export type LlamaServerImageToPromptRequest = {
  image_name: string;
  instruction?: string;
  model?: string | null;
  max_tokens?: number;
  task_id?: string | null;
};

export type LlamaServerImageToPromptResponse = {
  prompt: string;
  error?: string | null;
};

// --- RTK Query API ---

export const llamaServerApi = api.injectEndpoints({
  endpoints: (build) => ({
    // Configuration
    getLlamaServerConfig: build.query<LlamaServerConfig, void>({
      query: () => ({
        url: buildLlamaServerUrl('config'),
        method: 'GET',
      }),
      providesTags: ['AppConfig', 'FetchOnReconnect'],
    }),
    updateLlamaServerConfig: build.mutation<LlamaServerConfig, LlamaServerConfigUpdate>({
      query: (body) => ({
        url: buildLlamaServerUrl('config'),
        body,
        method: 'PATCH',
      }),
      invalidatesTags: ['AppConfig', 'FetchOnReconnect'],
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        try {
          const { data } = await queryFulfilled;
          dispatch(llamaServerApi.util.upsertQueryData('getLlamaServerConfig', undefined, data));
        } catch {
          // no-op
        }
      },
    }),

    // Test connection
    testLlamaServerConnection: build.mutation<LlamaServerTestResult, void>({
      query: () => ({
        url: buildLlamaServerUrl('test'),
        method: 'POST',
      }),
    }),

    // List models
    listLlamaServerModels: build.query<string[], void>({
      query: () => ({
        url: buildLlamaServerUrl('models'),
        method: 'GET',
      }),
    }),

    // Expand prompt via external llama server
    expandPromptLlamaServer: build.mutation<LlamaServerExpandPromptResponse, LlamaServerExpandPromptRequest>({
      query: (body) => ({
        url: buildLlamaServerUrl('expand-prompt'),
        body,
        method: 'POST',
      }),
    }),

    // Image-to-prompt via external llama server
    imageToPromptLlamaServer: build.mutation<LlamaServerImageToPromptResponse, LlamaServerImageToPromptRequest>({
      query: (body) => ({
        url: buildLlamaServerUrl('image-to-prompt'),
        body,
        method: 'POST',
      }),
    }),
  }),
});

export const {
  useGetLlamaServerConfigQuery,
  useUpdateLlamaServerConfigMutation,
  useTestLlamaServerConnectionMutation,
  useListLlamaServerModelsQuery,
  useLazyListLlamaServerModelsQuery,
  useExpandPromptLlamaServerMutation,
  useImageToPromptLlamaServerMutation,
} = llamaServerApi;
