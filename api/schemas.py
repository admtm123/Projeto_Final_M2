"""Contratos validados de entrada e saída da API imobiliária."""

from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


# Tipos estritos impedem que o Pydantic converta texto em número silenciosamente.
NumericFeature = Union[StrictInt, StrictFloat]

# As faixas refletem os dados e o tratamento usados para treinar o modelo v1.
BedroomCount = Annotated[NumericFeature, Field(ge=0, le=5.5)]
BathroomCount = Annotated[NumericFeature, Field(ge=0, le=8)]
LivingArea = Annotated[NumericFeature, Field(ge=290, le=13_540)]
LotArea = Annotated[NumericFeature, Field(ge=520, le=1_651_359)]
FloorCount = Annotated[NumericFeature, Field(ge=1, le=3.5)]
BinaryFlag = Annotated[StrictInt, Field(ge=0, le=1)]
ViewRating = Annotated[StrictInt, Field(ge=0, le=4)]
ConditionRating = Annotated[StrictInt, Field(ge=1, le=5)]
GradeRating = Annotated[StrictInt, Field(ge=1, le=13)]
BasementArea = Annotated[NumericFeature, Field(ge=0, le=4_820)]
PropertyAge = Annotated[StrictInt, Field(ge=0, le=115)]
Latitude = Annotated[NumericFeature, Field(ge=47.16, le=47.78)]
Longitude = Annotated[NumericFeature, Field(ge=-122.52, le=-121.31)]


class PredictionRequest(BaseModel):
    """Contrato de entrada: as 23 features esperadas pelo modelo v1."""

    # Campos desconhecidos são erros para evitar dados enviados por engano.
    model_config = ConfigDict(extra="forbid", strict=True)

    bedrooms: BedroomCount = Field(description="Quartos; limite após winsorização do treino.")
    bathrooms: BathroomCount = Field(description="Banheiros, entre 0 e 8.")
    sqft_living: LivingArea = Field(description="Área habitável em pés quadrados.")
    sqft_lot: LotArea = Field(description="Área do terreno em pés quadrados.")
    floors: FloorCount = Field(description="Andares; aceita valores fracionários do conjunto.")
    waterfront: BinaryFlag = Field(description="Indicador binário de imóvel à beira d'água.")
    view: ViewRating = Field(description="Classificação da vista, de 0 a 4.")
    condition: ConditionRating = Field(description="Condição do imóvel, de 1 a 5.")
    grade: GradeRating = Field(description="Classificação construtiva, de 1 a 13.")
    sqft_basement: BasementArea = Field(description="Área do porão em pés quadrados.")
    idade_imovel: PropertyAge = Field(
        description="Idade do imóvel em anos, no domínio do treino."
    )
    foi_reformado: BinaryFlag = Field(description="Indicador binário de reforma.")
    lat: Latitude = Field(
        description="Latitude dentro da área coberta pelo conjunto de treino."
    )
    long: Longitude = Field(
        description="Longitude dentro da área coberta pelo conjunto de treino."
    )
    zipcode_faixa_faixa_1: BinaryFlag = Field(description="Indicador binário da faixa 1 de CEP.")
    zipcode_faixa_faixa_2: BinaryFlag = Field(description="Indicador binário da faixa 2 de CEP.")
    zipcode_faixa_faixa_3: BinaryFlag = Field(description="Indicador binário da faixa 3 de CEP.")
    zipcode_faixa_faixa_4: BinaryFlag = Field(description="Indicador binário da faixa 4 de CEP.")
    zipcode_faixa_faixa_5: BinaryFlag = Field(description="Indicador binário da faixa 5 de CEP.")
    zipcode_faixa_faixa_6: BinaryFlag = Field(description="Indicador binário da faixa 6 de CEP.")
    zipcode_faixa_faixa_7: BinaryFlag = Field(description="Indicador binário da faixa 7 de CEP.")
    zipcode_faixa_faixa_8: BinaryFlag = Field(description="Indicador binário da faixa 8 de CEP.")
    zipcode_faixa_faixa_9: BinaryFlag = Field(description="Indicador binário da faixa 9 de CEP.")


class PredictionResponse(BaseModel):
    """Contrato de saída com o preço previsto e a versão do modelo."""

    # A resposta também é estrita: o preço precisa ser positivo e a versão curta.
    model_config = ConfigDict(extra="forbid", strict=True)

    preco_estimado: Annotated[StrictFloat, Field(gt=0)] = Field(
        description="Preço estimado; deve ser positivo."
    )
    versao_modelo: str = Field(
        min_length=1,
        max_length=20,
        description="Identificador da versão do modelo.",
    )
