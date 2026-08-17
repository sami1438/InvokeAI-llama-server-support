import {
  Button,
  Combobox,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  IconButton,
  Input,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Switch,
  Text,
  useToast,
} from '@invoke-ai/ui-library';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiArrowClockwiseBold, PiWarningBold } from 'react-icons/pi';
import {
  useGetLlamaServerConfigQuery,
  useUpdateLlamaServerConfigMutation,
  useTestLlamaServerConnectionMutation,
  useLazyListLlamaServerModelsQuery,
} from 'services/api/endpoints/llamaServer';

export const SettingsExternalLlamaServer = memo(() => {
  const { t } = useTranslation();
  const toast = useToast();
  const isAdmin = useIsAdmin();

  const { data: config } = useGetLlamaServerConfigQuery();
  const [updateConfig, { isLoading: isUpdating }] = useUpdateLlamaServerConfigMutation();
  const [testConnection, { isLoading: isTesting }] = useTestLlamaServerConnectionMutation();
  const [listModels, { data: availableModels, isLoading: isLoadingModels }] = useLazyListLlamaServerModelsQuery();

  // Local form state
  const [enabled, setEnabled] = useState(false);
  const [url, setUrl] = useState('http://127.0.0.1:8080');
  const [apiKey, setApiKey] = useState('');
  const [textModel, setTextModel] = useState('');
  const [visionModel, setVisionModel] = useState('');
  const [timeout, setTimeout_] = useState(120);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [temperature, setTemperature] = useState(0.7);

  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  // Sync local state from server config
  useEffect(() => {
    if (config) {
      setEnabled(config.enabled);
      setUrl(config.url ?? 'http://127.0.0.1:8080');
      setApiKey(''); // Don't populate API key for security
      setTextModel(config.text_model ?? '');
      setVisionModel(config.vision_model ?? '');
      setTimeout_(config.timeout);
      setMaxTokens(config.max_tokens);
      setTemperature(config.temperature);
    }
  }, [config]);

  const handleSave = useCallback(async () => {
    try {
      await updateConfig({
        enabled,
        url: url.trim() || null,
        api_key: apiKey.trim() || null,
        text_model: textModel.trim() || null,
        vision_model: visionModel.trim() || null,
        timeout,
        max_tokens: maxTokens,
        temperature,
      }).unwrap();
      toast({
        title: t('common.saved'),
        status: 'success',
        duration: 2000,
      });
    } catch {
      toast({
        title: t('common.error'),
        description: 'Failed to save llama-server configuration',
        status: 'error',
      });
    }
  }, [enabled, url, apiKey, textModel, visionModel, timeout, maxTokens, temperature, updateConfig, toast, t]);

  const handleTestConnection = useCallback(async () => {
    setTestResult(null);
    try {
      // Save first so the test uses current URL
      await updateConfig({
        enabled: true,
        url: url.trim() || null,
        api_key: apiKey.trim() || null,
        text_model: textModel.trim() || null,
        vision_model: visionModel.trim() || null,
        timeout,
        max_tokens: maxTokens,
        temperature,
      }).unwrap();

      const result = await testConnection().unwrap();
      if (result.ok) {
        setTestResult({ ok: true, message: t('prompt.llamaServerConnectionSuccess') });
      } else {
        setTestResult({ ok: false, message: result.error ?? t('prompt.llamaServerConnectionFailed') });
      }
    } catch {
      setTestResult({ ok: false, message: t('prompt.llamaServerConnectionFailed') });
    }
  }, [url, apiKey, textModel, visionModel, timeout, maxTokens, temperature, updateConfig, testConnection, t]);

  const handleRefreshModels = useCallback(async () => {
    try {
      await listModels().unwrap();
    } catch {
      toast({
        title: t('common.error'),
        description: 'Failed to list models from llama-server',
        status: 'error',
      });
    }
  }, [listModels, toast]);

  const modelOptions = (availableModels ?? []).map((m) => ({ label: m, value: m }));

  if (!isAdmin) return null;

  return (
    <Flex flexDir="column" gap={3}>
      <FormControl orientation="vertical">
        <FormLabel>{t('prompt.llamaServerEnabled')}</FormLabel>
        <Switch isChecked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
      </FormControl>

      {enabled && (
        <>
          <FormControl orientation="vertical">
            <FormLabel>{t('prompt.llamaServerUrl')}</FormLabel>
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="http://127.0.0.1:8080"
              size="sm"
              fontFamily="monospace"
            />
          </FormControl>

          <FormControl orientation="vertical">
            <FormLabel>{t('prompt.llamaServerApiKey')}</FormLabel>
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={config?.api_key_configured ? '••••••••' : t('common.optional')}
              size="sm"
            />
            {config?.api_key_configured && (
              <FormHelperText>An API key is currently configured. Leave blank to keep it unchanged.</FormHelperText>
            )}
          </FormControl>

          <FormControl orientation="vertical">
            <FormLabel>{t('prompt.llamaServerTextModel')}</FormLabel>
            <Flex gap={2} alignItems="center">
              <Combobox
                value={textModel ? { label: textModel, value: textModel } : null}
                options={modelOptions}
                onChange={(option) => setTextModel(option?.value ?? '')}
                placeholder="Model name"
                isClearable
                isSearchable
                size="sm"
              />
              <Input
                value={textModel}
                onChange={(e) => setTextModel(e.target.value)}
                placeholder="Enter model name manually"
                size="sm"
                flex={1}
              />
            </Flex>
          </FormControl>

          <FormControl orientation="vertical">
            <FormLabel>{t('prompt.llamaServerVisionModel')}</FormLabel>
            <Flex gap={2} alignItems="center">
              <Combobox
                value={visionModel ? { label: visionModel, value: visionModel } : null}
                options={modelOptions}
                onChange={(option) => setVisionModel(option?.value ?? '')}
                placeholder="Vision model name"
                isClearable
                isSearchable
                size="sm"
              />
              <Input
                value={visionModel}
                onChange={(e) => setVisionModel(e.target.value)}
                placeholder="Enter model name manually"
                size="sm"
                flex={1}
              />
            </Flex>
          </FormControl>

          <FormControl orientation="vertical">
            <FormLabel>{t('prompt.llamaServerTimeout')}</FormLabel>
            <NumberInput min={5} max={600} step={10} value={timeout} onChange={(val) => setTimeout_(val)} size="sm">
              <NumberInputField />
              <NumberInputStepper />
            </NumberInput>
          </FormControl>

          <FormControl orientation="vertical">
            <FormLabel>{t('prompt.llamaServerMaxTokens')}</FormLabel>
            <NumberInput min={1} max={8192} step={128} value={maxTokens} onChange={(val) => setMaxTokens(val)} size="sm">
              <NumberInputField />
              <NumberInputStepper />
            </NumberInput>
          </FormControl>

          <FormControl orientation="vertical">
            <FormLabel>{t('prompt.llamaServerTemperature')}</FormLabel>
            <NumberInput
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(val) => setTemperature(val)}
              precision={1}
              size="sm"
            >
              <NumberInputField />
              <NumberInputStepper />
            </NumberInput>
          </FormControl>

          <Flex gap={2}>
            <Button
              size="sm"
              variant="outline"
              onClick={handleTestConnection}
              isLoading={isTesting}
              flex={1}
            >
              {t('prompt.llamaServerTestConnection')}
            </Button>
            <IconButton
              size="sm"
              variant="outline"
              aria-label={t('prompt.llamaServerRefreshModels')}
              icon={<PiArrowClockwiseBold />}
              onClick={handleRefreshModels}
              isLoading={isLoadingModels}
            />
          </Flex>

          {testResult && (
            <Flex gap={2} alignItems="center" fontSize="sm" color={testResult.ok ? 'success.500' : 'error.500'}>
              {testResult.ok ? <PiCheckBold /> : <PiWarningBold />}
              <Text>{testResult.message}</Text>
            </Flex>
          )}

          <Button
            size="sm"
            colorScheme="invokeBlue"
            onClick={handleSave}
            isLoading={isUpdating}
          >
            {t('common.save')}
          </Button>
        </>
      )}
    </Flex>
  );
});

SettingsExternalLlamaServer.displayName = 'SettingsExternalLlamaServer';
