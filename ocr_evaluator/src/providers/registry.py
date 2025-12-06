"""
Provider registry for managing OCR providers.
Allows dynamic registration and discovery of providers.
"""

from typing import Dict, Type, Optional, List, Any
from pathlib import Path
import yaml

from .base import OCRProvider, OCRResult
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ProviderRegistry:
    """
    Registry for OCR providers.
    Manages provider registration, instantiation, and configuration.
    """
    
    _providers: Dict[str, Type[OCRProvider]] = {}
    _instances: Dict[str, OCRProvider] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a provider class.
        
        Usage:
            @ProviderRegistry.register("openai")
            class OpenAIVisionProvider(OCRProvider):
                ...
        """
        def decorator(provider_class: Type[OCRProvider]):
            cls._providers[name.lower()] = provider_class
            logger.info(f"Registered provider: {name}")
            return provider_class
        return decorator
    
    @classmethod
    def get_provider_class(cls, name: str) -> Optional[Type[OCRProvider]]:
        """Get a provider class by name."""
        return cls._providers.get(name.lower())
    
    @classmethod
    def get_provider(
        cls, 
        name: str, 
        config: Dict[str, Any] = None,
        force_new: bool = False
    ) -> OCRProvider:
        """
        Get or create a provider instance.
        
        Args:
            name: Provider name
            config: Provider configuration
            force_new: Create new instance even if one exists
        
        Returns:
            Provider instance
        
        Raises:
            ValueError if provider not found
        """
        name_lower = name.lower()
        
        # Return existing instance if available
        if not force_new and name_lower in cls._instances:
            return cls._instances[name_lower]
        
        # Get provider class
        provider_class = cls._providers.get(name_lower)
        if provider_class is None:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Provider '{name}' not found. Available: {available}"
            )
        
        # Create instance
        config = config or {}
        instance = provider_class(config)
        
        # Cache instance
        cls._instances[name_lower] = instance
        logger.info(f"Created provider instance: {name}")
        
        return instance
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def load_from_config(cls, config_path: Path) -> Dict[str, OCRProvider]:
        """
        Load and instantiate providers from configuration file.
        
        Args:
            config_path: Path to providers.yaml
        
        Returns:
            Dictionary of provider name -> instance
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        providers = {}
        provider_configs = config.get('providers', {})
        
        for name, provider_config in provider_configs.items():
            if not provider_config.get('enabled', True):
                logger.debug(f"Provider {name} is disabled, skipping")
                continue
            
            try:
                provider = cls.get_provider(name, provider_config)
                providers[name] = provider
            except ValueError as e:
                logger.warning(f"Could not load provider {name}: {e}")
        
        return providers
    
    @classmethod
    def clear_instances(cls):
        """Clear all cached provider instances."""
        cls._instances.clear()
        logger.debug("Cleared all provider instances")
    
    @classmethod
    def unregister(cls, name: str):
        """Unregister a provider."""
        name_lower = name.lower()
        if name_lower in cls._providers:
            del cls._providers[name_lower]
        if name_lower in cls._instances:
            del cls._instances[name_lower]
        logger.info(f"Unregistered provider: {name}")


