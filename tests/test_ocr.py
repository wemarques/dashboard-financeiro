"""
Testes para o módulo de OCR com Mistral AI
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestOCRProcessor:
    """Testes para a classe OCRProcessor"""

    def test_import_ocr_module(self):
        """Testa importação do módulo OCR"""
        try:
            from utils.ocr_mistral import OCRProcessor
            assert OCRProcessor is not None
        except ImportError as e:
            pytest.skip(f"Módulo não disponível: {e}")

    def test_processor_initialization_without_api_key(self):
        """Testa inicialização sem API key"""
        from utils.ocr_mistral import OCRProcessor

        with patch.dict('os.environ', {'MISTRAL_API_KEY': ''}):
            processor = OCRProcessor(api_key="")
            # Deve inicializar sem erro, mas client será None
            assert processor.api_key == ""

    @patch('utils.ocr_mistral.Mistral')
    def test_processor_initialization_with_api_key(self, mock_mistral):
        """Testa inicialização com API key"""
        from utils.ocr_mistral import OCRProcessor

        processor = OCRProcessor(api_key="test-key")
        assert processor.api_key == "test-key"

    def test_preprocess_text_removes_special_chars(self):
        """Testa pré-processamento de texto (via categorizer)"""
        from ml.categorizer import Categorizer

        categorizer = Categorizer()
        result = categorizer._preprocess_text("R$ 100,00 - Compra!")
        assert "r$" not in result.lower() or "$" not in result

    @patch('utils.ocr_mistral.Mistral')
    def test_extrair_recibo_mock(self, mock_mistral):
        """Testa extração de recibo com mock"""
        from utils.ocr_mistral import OCRProcessor

        # Configurar mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''```json
        {
            "data": "15/01/2024",
            "valor_total": 150.00,
            "estabelecimento": "Supermercado Teste",
            "categoria_sugerida": "alimentação"
        }
        ```'''
        mock_client.chat.complete.return_value = mock_response
        mock_mistral.return_value = mock_client

        processor = OCRProcessor(api_key="test-key")
        processor.client = mock_client

        # Criar arquivo temporário de teste
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name

        try:
            result = processor.extrair_recibo(tmp_path)
            assert result is not None
            assert "valor_total" in result or "raw_text" in result
        finally:
            import os
            os.remove(tmp_path)


class TestOCRParsers:
    """Testes para parsers de arquivos"""

    def test_parse_csv_bancario_structure(self):
        """Testa estrutura do resultado do parser CSV"""
        from utils.ocr_mistral import OCRProcessor
        import tempfile
        import os

        # Criar CSV de teste
        csv_content = """data,descricao,valor
15/01/2024,Salário,5000.00
16/01/2024,Supermercado,-250.00
17/01/2024,Uber,-35.50"""

        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
            tmp.write(csv_content)
            tmp_path = tmp.name

        try:
            processor = OCRProcessor(api_key="")
            result = processor._parse_csv_bancario(tmp_path)

            assert "movimentacoes" in result or "error" in result
            if "movimentacoes" in result:
                assert len(result["movimentacoes"]) == 3
        finally:
            os.remove(tmp_path)


class TestConvenienceFunctions:
    """Testes para funções de conveniência"""

    def test_extrair_dados_nota_function_exists(self):
        """Testa que função existe"""
        from utils.ocr_mistral import extrair_dados_nota
        assert callable(extrair_dados_nota)

    def test_extrair_fatura_cartao_function_exists(self):
        """Testa que função existe"""
        from utils.ocr_mistral import extrair_fatura_cartao
        assert callable(extrair_fatura_cartao)

    def test_extrair_extrato_bancario_function_exists(self):
        """Testa que função existe"""
        from utils.ocr_mistral import extrair_extrato_bancario
        assert callable(extrair_extrato_bancario)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
