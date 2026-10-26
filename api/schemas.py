"""Schemas de entrada e saída da API de previsão imobiliária."""

from typing import Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


FeatureNumber = Union[StrictInt, StrictFloat]


class PredictionRequest(BaseModel):
    """Dados de um imóvel no formato esperado pelo modelo v1."""

    model_config = ConfigDict(extra="forbid", strict=True)

    bedrooms: FeatureNumber = Field(..., description="Número de quartos.")
    bathrooms: FeatureNumber = Field(..., description="Número de banheiros.")
    sqft_living: FeatureNumber = Field(..., description="Área habitável.")
    sqft_lot: FeatureNumber = Field(..., description="Área do terreno.")
    floors: FeatureNumber = Field(..., description="Número de andares.")
    waterfront: FeatureNumber = Field(..., description="Indicador de imóvel à beira d'água.")
    view: FeatureNumber = Field(..., description="Classificação da vista.")
    condition: FeatureNumber = Field(..., description="Condição do imóvel.")
    grade: FeatureNumber = Field(..., description="Classificação construtiva.")
    sqft_basement: FeatureNumber = Field(..., description="Área do porão.")
    idade_imovel: FeatureNumber = Field(..., description="Idade do imóvel.")
    foi_reformado: FeatureNumber = Field(..., description="Indicador de reforma.")
    lat: FeatureNumber = Field(..., description="Latitude.")
    long: FeatureNumber = Field(..., description="Longitude.")
    zipcode_faixa_faixa_1: FeatureNumber = Field(..., description="Indicador da faixa 1 de CEP.")
    zipcode_faixa_faixa_2: FeatureNumber = Field(..., description="Indicador da faixa 2 de CEP.")
    zipcode_faixa_faixa_3: FeatureNumber = Field(..., description="Indicador da faixa 3 de CEP.")
    zipcode_faixa_faixa_4: FeatureNumber = Field(..., description="Indicador da faixa 4 de CEP.")
    zipcode_faixa_faixa_5: FeatureNumber = Field(..., description="Indicador da faixa 5 de CEP.")
    zipcode_faixa_faixa_6: FeatureNumber = Field(..., description="Indicador da faixa 6 de CEP.")
    zipcode_faixa_faixa_7: FeatureNumber = Field(..., description="Indicador da faixa 7 de CEP.")
    zipcode_faixa_faixa_8: FeatureNumber = Field(..., description="Indicador da faixa 8 de CEP.")
    zipcode_faixa_faixa_9: FeatureNumber = Field(..., description="Indicador da faixa 9 de CEP.")


class PredictionResponse(BaseModel):
    """Resposta padronizada da API com a previsão e a versão do modelo."""

    model_config = ConfigDict(extra="forbid", strict=True)

    preco_estimado: StrictFloat = Field(..., description="Preço estimado pelo modelo.")
    versao_modelo: str = Field(..., description="Versão do modelo usado na previsão.")
