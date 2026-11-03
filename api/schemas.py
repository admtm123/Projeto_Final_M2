"""Contratos de dados da API imobiliária, validados automaticamente pelo Pydantic.

Os modelos deste módulo descrevem o que a API aceita e o que devolve. O FastAPI
usa essas definições para validar requisições e documentar os endpoints no Swagger.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


# Tipos estritos impedem conversões implícitas: por exemplo, "3" como texto não
# é aceito como número. Essa união permite inteiros ou decimais onde o domínio
# da feature aceita ambos.
NumericFeature = Union[StrictInt, StrictFloat]

# Tipos reutilizáveis para expressar as restrições de cada grupo de features.
# Os limites refletem o domínio dos dados usados no treino da versão 1; rejeitar
# valores fora dessas faixas evita enviar ao modelo entradas não contempladas.
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
    """Contrato JSON de entrada da previsão, com as 23 features do modelo v1."""

    # A validação estrita rejeita tipos incompatíveis sem conversão automática.
    # extra="forbid" faz com que campos não declarados também sejam rejeitados,
    # ajudando a detectar erros de digitação e payloads fora do contrato.
    model_config = ConfigDict(extra="forbid", strict=True)

    # As descrições aparecem na documentação OpenAPI/Swagger e ajudam quem
    # consome a API a informar unidades, escalas e significado dos campos.
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
    """Formato padronizado da resposta de uma previsão bem-sucedida."""

    # Restringe a resposta aos campos declarados e evita conversões de tipos.
    model_config = ConfigDict(extra="forbid", strict=True)

    # O preço é numérico (não uma string formatada como moeda) e deve ser maior
    # que zero. O serviço arredonda o valor previsto antes de criar a resposta.
    preco_estimado: Annotated[StrictFloat, Field(gt=0)] = Field(
        description="Preço estimado; deve ser positivo."
    )
    # Limites de tamanho mantêm o identificador da versão presente e compacto.
    versao_modelo: str = Field(
        min_length=1,
        max_length=20,
        description="Identificador da versão do modelo.",
    )


class HealthResponse(BaseModel):
    """Formato da resposta de monitoramento do serviço e dos artefatos."""

    # Define exatamente as propriedades que o endpoint /health pode devolver.
    model_config = ConfigDict(extra="forbid", strict=True)

    # "degraded" indica que o serviço respondeu, mas um ou mais artefatos
    # necessários à inferência não estão disponíveis em memória.
    status: Literal["ok", "degraded"] = Field(
        description="Estado geral; degraded indica ausência de artefato em memória."
    )
    service_active: bool = Field(description="Indica que a API respondeu à checagem.")
    model_loaded: bool = Field(description="Indica que o modelo está disponível em RAM.")
    scaler_loaded: bool = Field(description="Indica que o scaler está disponível em RAM.")
