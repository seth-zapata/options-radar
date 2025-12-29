"""Data clients for OptionsRadar."""

from backend.data.aggregator import AggregatedOptionData, DataAggregator
from backend.data.alpaca_account import (
    AccountInfo,
    AlpacaAccountClient,
    PortfolioSummary,
    Position,
)
from backend.data.alpaca_client import AlpacaOptionsClient, ConnectionState
from backend.data.alpaca_rest import AlpacaRestClient, BarData, LatestQuote
from backend.data.eodhd_client import (
    EODHDClient,
    OptionContract,
    OptionsChain,
    OptionsIndicators,
)
from backend.data.finnhub_client import FinnhubClient, NewsSentiment, CompanyNews, SocialSentiment
from backend.data.market_hours import MarketStatus, check_market_hours, CT, ET
from backend.data.mock_data import MockDataGenerator
from backend.data.orats_client import ORATSClient
from backend.data.quiver_client import (
    QuiverClient,
    CongressTrade,
    InsiderTrade,
    PoliticalSentiment,
    WSBSentiment,
)
from backend.data.sentiment_aggregator import (
    CombinedSentiment,
    SentimentAggregator,
)
from backend.data.staleness import (
    FreshnessLevel,
    FreshnessReport,
    StalenessChecker,
    StalenessThresholds,
)
from backend.data.subscription_manager import SubscriptionManager
from backend.data.technicals import (
    TechnicalAnalyzer,
    TechnicalIndicators,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)

__all__ = [
    "AccountInfo",
    "AggregatedOptionData",
    "AlpacaAccountClient",
    "AlpacaOptionsClient",
    "AlpacaRestClient",
    "BarData",
    "CombinedSentiment",
    "CompanyNews",
    "CongressTrade",
    "ConnectionState",
    "CT",
    "DataAggregator",
    "EODHDClient",
    "ET",
    "FinnhubClient",
    "FreshnessLevel",
    "FreshnessReport",
    "InsiderTrade",
    "LatestQuote",
    "MarketStatus",
    "MockDataGenerator",
    "NewsSentiment",
    "OptionContract",
    "OptionsChain",
    "OptionsIndicators",
    "ORATSClient",
    "PoliticalSentiment",
    "PortfolioSummary",
    "Position",
    "QuiverClient",
    "SentimentAggregator",
    "SocialSentiment",
    "StalenessChecker",
    "StalenessThresholds",
    "SubscriptionManager",
    "TechnicalAnalyzer",
    "TechnicalIndicators",
    "WSBSentiment",
    "calculate_bollinger_bands",
    "calculate_ema",
    "calculate_macd",
    "calculate_rsi",
    "calculate_sma",
    "check_market_hours",
]
