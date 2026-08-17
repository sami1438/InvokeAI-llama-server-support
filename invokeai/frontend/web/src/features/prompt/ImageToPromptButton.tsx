import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Image,
  Input,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  spinAnimation,
  Text,
  Textarea,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { positivePromptChanged, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { LLMTaskProgressDisplay } from 'features/prompt/LLMTaskProgressDisplay';
import { setPromptUndo } from 'features/prompt/promptUndo';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';
import {
  useImageToPromptLlamaServerMutation,
  useGetLlamaServerConfigQuery,
} from 'services/api/endpoints/llamaServer';
import { useImageToPromptMutation } from 'services/api/endpoints/utilities';
import { useLlavaModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig, ImageDTO } from 'services/api/types';
import { clearLLMTaskState } from 'services/events/stores';
import { v4 as uuidv4 } from 'uuid';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

type Props = {
  droppedImage?: ImageDTO;
  onClearDroppedImage?: () => void;
};

type VisionProvider = 'invokeai' | 'external';

export const ImageToPromptButton = memo(({ droppedImage, onClearDroppedImage }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const currentPrompt = useAppSelector(selectPositivePrompt);
  const [modelConfigs] = useLlavaModels();
  const popover = useDisclosure(false);
  const [selectedModel, setSelectedModel] = useState<AnyModelConfig | undefined>(undefined);
  const [uploadedImage, setUploadedImage] = useState<ImageDTO | undefined>(undefined);
  const [taskId, setTaskId] = useState<string | null>(null);

  // Local InvokeAI vision
  const [imageToPrompt, { isLoading: isLocalLoading }] = useImageToPromptMutation();

  // External llama server
  const { data: llamaConfig } = useGetLlamaServerConfigQuery();
  const [imageToPromptLlamaServer, { isLoading: isExternalLoading }] = useImageToPromptLlamaServerMutation();

  const hasLocalModels = modelConfigs.length > 0;
  const isExternalEnabled = llamaConfig?.enabled && !!llamaConfig.url;

  const hasAnyProvider = hasLocalModels || isExternalEnabled;

  // Default to external if enabled and no local models, otherwise local
  const [visionProvider, setVisionProvider] = useState<VisionProvider>(
    isExternalEnabled && !hasLocalModels ? 'external' : 'invokeai'
  );

  const [instruction, setInstruction] = useState(
    'Describe this image in detail for use as an AI image generation prompt.'
  );

  // When a gallery image is dropped onto the prompt box, auto-open and set the image
  useEffect(() => {
    if (droppedImage) {
      setUploadedImage(droppedImage);
      popover.open();
      onClearDroppedImage?.();
    }
  }, [droppedImage, onClearDroppedImage, popover]);

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    allowMultiple: false,
    onUpload: setUploadedImage,
  });

  const isLoading = isLocalLoading || isExternalLoading;

  const handleModelChange = useCallback((model: AnyModelConfig) => {
    setSelectedModel(model);
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!uploadedImage) {
      return;
    }

    const newTaskId = uuidv4();
    setTaskId(newTaskId);

    try {
      if (visionProvider === 'external') {
        const result = await imageToPromptLlamaServer({
          image_name: uploadedImage.image_name,
          instruction,
          task_id: newTaskId,
        }).unwrap();
        if (result.prompt) {
          setPromptUndo(currentPrompt);
          dispatch(positivePromptChanged(result.prompt));
        }
      } else {
        if (!selectedModel) {
          return;
        }
        const result = await imageToPrompt({
          image_name: uploadedImage.image_name,
          model_key: selectedModel.key,
          instruction,
          task_id: newTaskId,
        }).unwrap();
        if (result.prompt) {
          setPromptUndo(currentPrompt);
          dispatch(positivePromptChanged(result.prompt));
        }
      }
      popover.close();
      setUploadedImage(undefined);
    } catch {
      // Error is handled by RTK Query
    } finally {
      clearLLMTaskState(newTaskId);
      setTaskId(null);
    }
  }, [
    visionProvider,
    uploadedImage,
    imageToPromptLlamaServer,
    imageToPrompt,
    instruction,
    selectedModel,
    dispatch,
    popover,
    currentPrompt,
  ]);

  const handleClose = useCallback(() => {
    popover.close();
    setUploadedImage(undefined);
  }, [popover]);

  const handleOpenModelManager = useCallback(() => {
    handleClose();
    navigationApi.switchToTab('models');
    setInstallModelsTabByName('starterModels');
  }, [handleClose]);

  const canGenerate = useMemo(() => {
    if (!uploadedImage) return false;
    if (visionProvider === 'external') return isExternalEnabled;
    return !!selectedModel;
  }, [uploadedImage, visionProvider, isExternalEnabled, selectedModel]);

  return (
    <Popover
      isOpen={popover.isOpen}
      onOpen={popover.open}
      onClose={handleClose}
      placement="left-start"
      isLazy
      closeOnBlur={false}
    >
      <PopoverTrigger>
        <span>
          <Tooltip label={hasAnyProvider ? t('prompt.imageToPrompt') : t('prompt.noVisionModelInstalledTitle')}>
            <IconButton
              size="sm"
              variant="promptOverlay"
              aria-label={t('prompt.imageToPrompt')}
              icon={<PiImageBold />}
              sx={isLoading ? loadingStyles : undefined}
              isDisabled={isLoading}
            />
          </Tooltip>
        </span>
      </PopoverTrigger>
      <Portal>
        <PopoverContent p={3} w={380}>
          <PopoverArrow />
          <PopoverBody p={0}>
            {hasAnyProvider ? (
              <Flex flexDir="column" gap={3}>
                <Text fontWeight="semibold" fontSize="sm">
                  {t('prompt.imageToPrompt')}
                </Text>

                {/* Provider selector - only show if both providers are available */}
                {hasLocalModels && isExternalEnabled && (
                  <FormControl orientation="vertical">
                    <FormLabel m={0} fontSize="xs">
                      {t('prompt.visionProvider')}
                    </FormLabel>
                    <Flex gap={2}>
                      <Button
                        size="xs"
                        variant={visionProvider === 'invokeai' ? 'solid' : 'outline'}
                        onClick={() => setVisionProvider('invokeai')}
                        flex={1}
                      >
                        {t('prompt.invokeAIModel')}
                      </Button>
                      <Button
                        size="xs"
                        variant={visionProvider === 'external' ? 'solid' : 'outline'}
                        onClick={() => setVisionProvider('external')}
                        flex={1}
                      >
                        {t('prompt.externalLlamaServer')}
                      </Button>
                    </Flex>
                  </FormControl>
                )}

                {/* Model picker for InvokeAI provider */}
                {visionProvider === 'invokeai' && (
                  <ModelPicker
                    pickerId="image-to-prompt-model"
                    modelConfigs={modelConfigs}
                    selectedModelConfig={selectedModel}
                    onChange={handleModelChange}
                    placeholder={t('prompt.selectVisionModel')}
                  />
                )}

                {/* External model name display */}
                {visionProvider === 'external' && llamaConfig?.vision_model && (
                  <Flex fontSize="xs" color="base.400">
                    {t('prompt.externalVisionModel')}: {llamaConfig.vision_model}
                  </Flex>
                )}

                {/* Instruction */}
                <FormControl orientation="vertical">
                  <FormLabel m={0} fontSize="xs">
                    {t('prompt.visionInstruction')}
                  </FormLabel>
                  <Textarea
                    value={instruction}
                    onChange={(e) => setInstruction(e.target.value)}
                    size="sm"
                    resize="vertical"
                    minH={12}
                    maxH={28}
                  />
                </FormControl>

                <Flex gap={2} alignItems="center">
                  <Button size="sm" variant="outline" flexGrow={1} {...getUploadButtonProps()}>
                    {uploadedImage ? t('prompt.changeImage') : t('prompt.uploadImage')}
                  </Button>
                  <input {...getUploadInputProps()} />
                  {uploadedImage && (
                    <Image
                      src={uploadedImage.image_url}
                      alt="Uploaded"
                      boxSize={10}
                      objectFit="cover"
                      borderRadius="base"
                    />
                  )}
                </Flex>
                {isLoading ? <LLMTaskProgressDisplay taskId={taskId} /> : null}
                <Button
                  size="sm"
                  colorScheme="invokeBlue"
                  onClick={handleGenerate}
                  isLoading={isLoading}
                  isDisabled={!canGenerate}
                >
                  {t('prompt.generatePrompt')}
                </Button>
              </Flex>
            ) : (
              <Flex flexDir="column" gap={3}>
                <Text fontWeight="semibold" fontSize="sm">
                  {t('prompt.noVisionModelInstalledTitle')}
                </Text>
                <Text fontSize="sm" color="base.300">
                  {t('prompt.noVisionModelInstalledDescription')}
                </Text>
                <Button size="sm" colorScheme="invokeBlue" onClick={handleOpenModelManager}>
                  {t('prompt.openModelManager')}
                </Button>
              </Flex>
            )}
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

ImageToPromptButton.displayName = 'ImageToPromptButton';
