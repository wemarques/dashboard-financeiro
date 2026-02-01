"""
Módulo de OCR usando Mistral AI para extração de dados de documentos financeiros
"""
import os
import json
import base64
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

try:
    from PIL import Image
    from pdf2image import convert_from_path
except ImportError:
    Image = None
    convert_from_path = None

try:
    from config import MISTRAL_API_KEY, MISTRAL_MODEL_SIMPLE, MISTRAL_MODEL_COMPLEX
    from utils.logger import get_logger, log_ocr_result
except ImportError:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL_SIMPLE = "pixtral-12b-latest"
    MISTRAL_MODEL_COMPLEX = "pixtral-large-latest"
    from logger import get_logger, log_ocr_result

logger = get_logger(__name__)


class OCRProcessor:
    """Processador de OCR usando Mistral AI"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or MISTRAL_API_KEY
        if not self.api_key:
            logger.warning("MISTRAL_API_KEY não configurada")

        if Mistral:
            self.client = Mistral(api_key=self.api_key) if self.api_key else None
        else:
            self.client = None
            logger.error("Biblioteca mistralai não instalada")

    def _image_to_base64(self, image_path: str) -> str:
        """Converte imagem para base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """Converte PDF para lista de imagens temporárias"""
        if not convert_from_path:
            raise ImportError("pdf2image não instalado. Execute: pip install pdf2image")

        images = convert_from_path(pdf_path, dpi=200)
        temp_paths = []

        for i, image in enumerate(images):
            temp_path = tempfile.mktemp(suffix=f"_page_{i}.jpg")
            image.save(temp_path, "JPEG", quality=95)
            temp_paths.append(temp_path)
            logger.debug(f"Página {i+1} do PDF convertida")

        return temp_paths

    def _call_mistral_ocr(self, image_base64: str, prompt: str, use_complex_model: bool = False) -> Optional[Dict]:
        """
        Faz chamada ao Mistral AI com imagem

        Args:
            image_base64: Imagem codificada em base64
            prompt: Prompt para extração de dados
            use_complex_model: Se True, usa modelo complexo (faturas/extratos).
                              Se False, usa modelo simples (recibos).
        """
        if not self.client:
            logger.error("Cliente Mistral não inicializado")
            return None

        model = MISTRAL_MODEL_COMPLEX if use_complex_model else MISTRAL_MODEL_SIMPLE
        logger.debug(f"Usando modelo Mistral: {model}")

        try:
            response = self.client.chat.complete(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            content = response.choices[0].message.content

            # Tentar extrair JSON da resposta
            try:
                # Procurar por blocos JSON na resposta
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()

                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Resposta não é JSON válido: {content[:200]}")
                return {"raw_text": content}

        except Exception as e:
            logger.error(f"Erro na chamada Mistral: {e}")
            return None

    def extrair_recibo(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extrai dados de um recibo/cupom fiscal

        Args:
            image_path: Caminho para a imagem do recibo

        Returns:
            Dicionário com dados extraídos ou None em caso de erro
        """
        logger.info(f"Processando recibo: {image_path}")

        prompt = """Analise esta imagem de recibo/cupom fiscal e extraia os dados em formato JSON:
{
    "data": "DD/MM/AAAA",
    "valor_total": 0.00,
    "estabelecimento": "nome do estabelecimento",
    "cnpj": "00.000.000/0000-00",
    "endereco": "endereço se visível",
    "forma_pagamento": "dinheiro/cartão/pix",
    "categoria_sugerida": "alimentação|transporte|saúde|lazer|compras|outros",
    "itens": [
        {"descricao": "descrição do item", "valor": 0.00, "quantidade": 1}
    ]
}

Se algum campo não estiver visível, use null. Retorne APENAS o JSON, sem explicações."""

        image_b64 = self._image_to_base64(image_path)
        # Recibos simples usam modelo econômico
        result = self._call_mistral_ocr(image_b64, prompt, use_complex_model=False)

        if result and "raw_text" not in result:
            log_ocr_result(logger, True, Path(image_path).name, len(result.get("itens", [])))
        else:
            log_ocr_result(logger, False, Path(image_path).name)

        return result

    def extrair_fatura_cartao(self, file_path: str) -> Dict[str, Any]:
        """
        Extrai transações de uma fatura de cartão de crédito

        Args:
            file_path: Caminho para o PDF ou imagem da fatura

        Returns:
            Dicionário com dados da fatura e lista de transações
        """
        logger.info(f"Processando fatura de cartão: {file_path}")

        prompt = """Analise esta fatura de cartão de crédito e extraia TODAS as transações em JSON:
{
    "banco": "nome do banco/bandeira do cartão",
    "mes_referencia": "MM/AAAA",
    "vencimento": "DD/MM/AAAA",
    "valor_total": 0.00,
    "valor_minimo": 0.00,
    "transacoes": [
        {
            "data": "DD/MM/AAAA",
            "descricao": "descrição da compra",
            "valor": 0.00,
            "parcela": "1/3 ou null se não for parcelado",
            "categoria_sugerida": "alimentação|transporte|saúde|lazer|delivery|jogos|assinaturas|compras|outros"
        }
    ]
}

IMPORTANTE: Extraia TODAS as transações visíveis. Retorne APENAS o JSON."""

        all_transactions = []
        fatura_info = {}

        # Se for PDF, converter para imagens
        if file_path.lower().endswith(".pdf"):
            image_paths = self._pdf_to_images(file_path)
        else:
            image_paths = [file_path]

        for i, img_path in enumerate(image_paths):
            logger.debug(f"Processando página {i+1}/{len(image_paths)}")
            image_b64 = self._image_to_base64(img_path)
            # Faturas de cartão usam modelo complexo para maior precisão
            result = self._call_mistral_ocr(image_b64, prompt, use_complex_model=True)

            if result and "raw_text" not in result:
                # Mesclar informações da fatura
                if not fatura_info:
                    fatura_info = {
                        "banco": result.get("banco"),
                        "mes_referencia": result.get("mes_referencia"),
                        "vencimento": result.get("vencimento"),
                        "valor_total": result.get("valor_total"),
                        "valor_minimo": result.get("valor_minimo")
                    }

                # Adicionar transações
                all_transactions.extend(result.get("transacoes", []))

            # Limpar arquivos temporários de PDF
            if file_path.lower().endswith(".pdf") and os.path.exists(img_path):
                os.remove(img_path)

        fatura_info["transacoes"] = all_transactions
        log_ocr_result(logger, bool(all_transactions), Path(file_path).name, len(all_transactions))

        return fatura_info

    def extrair_extrato_bancario(self, file_path: str) -> Dict[str, Any]:
        """
        Extrai movimentações de um extrato bancário

        Args:
            file_path: Caminho para PDF, imagem, OFX ou CSV do extrato

        Returns:
            Dicionário com dados do extrato e lista de movimentações
        """
        logger.info(f"Processando extrato bancário: {file_path}")

        # Tratar OFX (formato padrão de bancos)
        if file_path.lower().endswith(".ofx"):
            return self._parse_ofx(file_path)

        # Tratar CSV
        if file_path.lower().endswith(".csv"):
            return self._parse_csv_bancario(file_path)

        # PDF ou imagem - usar OCR
        prompt = """Analise este extrato bancário e extraia TODAS as movimentações em JSON:
{
    "banco": "nome do banco",
    "agencia": "número da agência",
    "conta": "número da conta",
    "periodo": {
        "inicio": "DD/MM/AAAA",
        "fim": "DD/MM/AAAA"
    },
    "saldo_inicial": 0.00,
    "saldo_final": 0.00,
    "movimentacoes": [
        {
            "data": "DD/MM/AAAA",
            "descricao": "descrição da movimentação",
            "valor": 0.00,
            "tipo": "credito ou debito",
            "categoria_sugerida": "salário|transferência|pix|boleto|saque|depósito|outros"
        }
    ]
}

IMPORTANTE: Valores de entrada são "credito", saídas são "debito". Retorne APENAS o JSON."""

        all_movimentacoes = []
        extrato_info = {}

        # Se for PDF, converter para imagens
        if file_path.lower().endswith(".pdf"):
            image_paths = self._pdf_to_images(file_path)
        else:
            image_paths = [file_path]

        for i, img_path in enumerate(image_paths):
            logger.debug(f"Processando página {i+1}/{len(image_paths)}")
            image_b64 = self._image_to_base64(img_path)
            # Extratos bancários usam modelo complexo para maior precisão
            result = self._call_mistral_ocr(image_b64, prompt, use_complex_model=True)

            if result and "raw_text" not in result:
                if not extrato_info:
                    extrato_info = {
                        "banco": result.get("banco"),
                        "agencia": result.get("agencia"),
                        "conta": result.get("conta"),
                        "periodo": result.get("periodo"),
                        "saldo_inicial": result.get("saldo_inicial"),
                        "saldo_final": result.get("saldo_final")
                    }

                all_movimentacoes.extend(result.get("movimentacoes", []))

            # Limpar temporários
            if file_path.lower().endswith(".pdf") and os.path.exists(img_path):
                os.remove(img_path)

        extrato_info["movimentacoes"] = all_movimentacoes
        log_ocr_result(logger, bool(all_movimentacoes), Path(file_path).name, len(all_movimentacoes))

        return extrato_info

    def _parse_ofx(self, file_path: str) -> Dict[str, Any]:
        """Parse de arquivo OFX (Open Financial Exchange)"""
        try:
            from ofxparse import OfxParser
        except ImportError:
            logger.error("ofxparse não instalado. Execute: pip install ofxparse")
            return {"error": "ofxparse não instalado"}

        try:
            with open(file_path, 'rb') as f:
                ofx = OfxParser.parse(f)

            movimentacoes = []
            for account in ofx.accounts:
                for transaction in account.statement.transactions:
                    tipo = "credito" if transaction.amount > 0 else "debito"
                    movimentacoes.append({
                        "data": transaction.date.strftime("%d/%m/%Y"),
                        "descricao": transaction.memo or transaction.payee or "Sem descrição",
                        "valor": abs(float(transaction.amount)),
                        "tipo": tipo,
                        "categoria_sugerida": "outros"
                    })

            logger.info(f"OFX processado: {len(movimentacoes)} movimentações")
            return {
                "banco": ofx.account.institution.organization if hasattr(ofx.account, 'institution') else "Desconhecido",
                "conta": ofx.account.account_id if hasattr(ofx.account, 'account_id') else "",
                "periodo": {
                    "inicio": ofx.account.statement.start_date.strftime("%d/%m/%Y") if hasattr(ofx.account.statement, 'start_date') else None,
                    "fim": ofx.account.statement.end_date.strftime("%d/%m/%Y") if hasattr(ofx.account.statement, 'end_date') else None
                },
                "saldo_inicial": None,
                "saldo_final": float(ofx.account.statement.balance) if hasattr(ofx.account.statement, 'balance') else None,
                "movimentacoes": movimentacoes
            }
        except Exception as e:
            logger.error(f"Erro ao processar OFX: {e}")
            return {"error": str(e)}

    def _parse_csv_bancario(self, file_path: str) -> Dict[str, Any]:
        """Parse de arquivo CSV de extrato bancário"""
        import pandas as pd

        try:
            # Tentar diferentes encodings e separadores
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for sep in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:
                            break
                    except:
                        continue
                else:
                    continue
                break

            # Normalizar nomes de colunas
            df.columns = df.columns.str.lower().str.strip()

            # Mapear colunas comuns
            col_mapping = {
                'data': ['data', 'date', 'dt', 'data_lancamento'],
                'descricao': ['descricao', 'description', 'historico', 'desc', 'lancamento'],
                'valor': ['valor', 'value', 'amount', 'vlr'],
                'tipo': ['tipo', 'type', 'd/c', 'natureza']
            }

            def find_column(options):
                for opt in options:
                    if opt in df.columns:
                        return opt
                return None

            data_col = find_column(col_mapping['data'])
            desc_col = find_column(col_mapping['descricao'])
            valor_col = find_column(col_mapping['valor'])

            movimentacoes = []
            for _, row in df.iterrows():
                valor = float(str(row.get(valor_col, 0)).replace(',', '.').replace('R$', '').strip())
                tipo = "credito" if valor > 0 else "debito"

                movimentacoes.append({
                    "data": str(row.get(data_col, "")),
                    "descricao": str(row.get(desc_col, "Sem descrição")),
                    "valor": abs(valor),
                    "tipo": tipo,
                    "categoria_sugerida": "outros"
                })

            logger.info(f"CSV processado: {len(movimentacoes)} movimentações")
            return {
                "banco": "Importado de CSV",
                "movimentacoes": movimentacoes
            }

        except Exception as e:
            logger.error(f"Erro ao processar CSV: {e}")
            return {"error": str(e)}


# === Funções de conveniência ===

def extrair_dados_nota(image_path: str) -> Optional[Dict[str, Any]]:
    """Função de conveniência para extrair dados de recibo"""
    processor = OCRProcessor()
    return processor.extrair_recibo(image_path)


def extrair_fatura_cartao(file_path: str) -> Dict[str, Any]:
    """Função de conveniência para extrair fatura de cartão"""
    processor = OCRProcessor()
    return processor.extrair_fatura_cartao(file_path)


def extrair_extrato_bancario(file_path: str) -> Dict[str, Any]:
    """Função de conveniência para extrair extrato bancário"""
    processor = OCRProcessor()
    return processor.extrair_extrato_bancario(file_path)
