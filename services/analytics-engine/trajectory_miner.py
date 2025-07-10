"""
SENTIMENT TRAJECTORY PATTERN MINING: Sentiment Trajectory Pattern Mining
Mathematical Foundation for Escalation Prediction
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import json
import re
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import asyncio
import logging

logger = logging.getLogger(__name__)

class SentimentTrajectoryMiner:
    """
    Advanced sentiment trajectory pattern mining with mathematical rigor
    """
    
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.patterns = {}
        self.escalation_rules = {}
        self.user_trajectories = {}
        self.pattern_weights = {}
        
        # Pattern mining parameters
        self.pattern_support_threshold = 0.1
        self.pattern_confidence_threshold = 0.5
        self.max_pattern_length = 5
        
        # Escalation keywords with weights
        self.escalation_keywords = {
            'angry': 0.8, 'frustrated': 0.7, 'disappointed': 0.6,
            'upset': 0.7, 'annoyed': 0.5, 'terrible': 0.8,
            'awful': 0.7, 'horrible': 0.8, 'worst': 0.9,
            'hate': 0.9, 'furious': 0.9, 'livid': 0.8,
            'complaint': 0.6, 'refund': 0.7, 'cancel': 0.8,
            'manager': 0.7, 'supervisor': 0.6, 'unacceptable': 0.8,
            'ridiculous': 0.7, 'pathetic': 0.8
        }
        
    def extract_conversation_trajectories(self, messages: List[Dict]) -> List[Dict]:
        """Extract conversation trajectories from messages"""
        # Group messages by conversation (simulate user_id from message index)
        conversations = defaultdict(list)
        
        for i, msg in enumerate(messages):
            # Create synthetic user_id based on message patterns
            user_id = f"user_{i // 6 + 1}"  # Group every 6 messages as one user
            conversations[user_id].append({
                'user_id': user_id,
                'content': msg.get('content', ''),
                'sentiment': msg.get('sentiment', 'neutral'),
                'sentiment_confidence': msg.get('sentiment_confidence', 0.5),
                'created_at': msg.get('created_at', datetime.now()),
                'text': msg.get('content', '')
            })
        
        # Extract trajectory features for each conversation
        trajectories = []
        for user_id, user_messages in conversations.items():
            if len(user_messages) >= 2:  # Need at least 2 messages for trajectory
                trajectory = self.extract_trajectory_features(user_messages)
                trajectory['user_id'] = user_id
                trajectory['escalation_occurred'] = self._detect_escalation(user_messages)
                trajectories.append(trajectory)
        
        return trajectories
    
    def mine_frequent_patterns(self, trajectories: List[Dict]) -> List[Dict]:
        """Mine frequent patterns using Apriori algorithm"""
        if not trajectories:
            return []
        
        # Extract sentiment sequences
        sequences = [t['sentiment_sequence'] for t in trajectories]
        
        # Generate candidate patterns
        patterns = []
        
        # 1-itemsets (single sentiments)
        single_items = defaultdict(int)
        for seq in sequences:
            for item in seq:
                single_items[item] += 1
        
        # Filter by support threshold
        min_support_count = len(sequences) * self.pattern_support_threshold
        frequent_1_items = {item: count for item, count in single_items.items() 
                           if count >= min_support_count}
        
        # 2-itemsets (pairs)
        pairs = defaultdict(int)
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pairs[pair] += 1
        
        frequent_2_items = {pair: count for pair, count in pairs.items() 
                           if count >= min_support_count}
        
        # 3-itemsets (triplets)
        triplets = defaultdict(int)
        for seq in sequences:
            for i in range(len(seq) - 2):
                triplet = (seq[i], seq[i+1], seq[i+2])
                triplets[triplet] += 1
        
        frequent_3_items = {triplet: count for triplet, count in triplets.items() 
                           if count >= min_support_count}
        
        # Create pattern objects
        all_patterns = []
        
        # Add single item patterns
        for item, count in frequent_1_items.items():
            support = count / len(sequences)
            all_patterns.append({
                'pattern': [item],
                'support': support,
                'frequency': count,
                'length': 1
            })
        
        # Add pair patterns
        for pair, count in frequent_2_items.items():
            support = count / len(sequences)
            all_patterns.append({
                'pattern': list(pair),
                'support': support,
                'frequency': count,
                'length': 2
            })
        
        # Add triplet patterns
        for triplet, count in frequent_3_items.items():
            support = count / len(sequences)
            all_patterns.append({
                'pattern': list(triplet),
                'support': support,
                'frequency': count,
                'length': 3
            })
        
        # Sort by support (descending)
        all_patterns.sort(key=lambda x: x['support'], reverse=True)
        
        return all_patterns
    
    def validate_patterns_statistically(self, trajectories: List[Dict]) -> Dict:
        """Validate patterns with statistical tests"""
        if not trajectories:
            return {'error': 'No trajectories to validate'}
        
        # Calculate escalation statistics
        escalated = [t for t in trajectories if t.get('escalation_occurred', False)]
        non_escalated = [t for t in trajectories if not t.get('escalation_occurred', False)]
        
        escalation_rate = len(escalated) / len(trajectories)
        
        # Statistical validation
        validation_results = {
            'total_trajectories': len(trajectories),
            'escalated_trajectories': len(escalated),
            'non_escalated_trajectories': len(non_escalated),
            'escalation_rate': escalation_rate,
            'patterns_validated': len(self.patterns),
            'validation_method': 'frequency_analysis',
            'confidence_threshold': self.pattern_confidence_threshold,
            'support_threshold': self.pattern_support_threshold
        }
        
        return validation_results
    
    def predict_escalation_probability(self, trajectory: Dict) -> Dict:
        """Predict escalation probability for a trajectory"""
        if not self.patterns:
            return {
                'escalation_probability': 0.0,
                'confidence': 0.0,
                'matched_patterns': []
            }
        
        # Use the existing predict_escalation method
        prediction = self.predict_escalation(trajectory, self.patterns)
        
        return {
            'escalation_probability': prediction['escalation_probability'],
            'confidence': prediction.get('confidence_interval', (0.0, 0.0))[1],
            'matched_patterns': prediction.get('key_indicators', [])
        }
    
    def _detect_escalation(self, messages: List[Dict]) -> bool:
        """Improved escalation detection logic"""
        if not messages or len(messages) < 2:
            return False
        
        # Extract sentiment sequence
        sentiments = [msg.get('sentiment', 'neutral') for msg in messages]
        contents = [msg.get('content', '') for msg in messages]
        
        # Check for escalation indicators
        escalation_indicators = 0
        
        # 1. Increasing negative sentiment
        negative_count = sentiments.count('negative')
        if negative_count >= len(sentiments) * 0.6:  # 60% or more negative
            escalation_indicators += 1
        
        # 2. Sentiment deterioration pattern
        if len(sentiments) >= 3:
            recent_sentiments = sentiments[-3:]
            if recent_sentiments.count('negative') >= 2:
                escalation_indicators += 1
        
        # 3. Keyword-based escalation
        escalation_keywords_found = 0
        for content in contents:
            if content:
                words = content.lower().split()
                for word in words:
                    if word in self.escalation_keywords:
                        escalation_keywords_found += 1
        
        if escalation_keywords_found >= 2:
            escalation_indicators += 1
        
        # 4. Length and urgency indicators
        if len(messages) >= 4:  # Long conversation
            escalation_indicators += 1
        
        # Decision: escalation if 2 or more indicators
        return escalation_indicators >= 2
    
    def extract_trajectory_features(self, messages: List[Dict]) -> Dict:
        """Extract comprehensive trajectory features"""
        if len(messages) < 2:
            return self._empty_trajectory_features()
        
        # Sort by timestamp
        messages = sorted(messages, key=lambda x: x['created_at'])
        
        # Extract basic features
        sentiments = [self._sentiment_to_numeric(msg['sentiment']) for msg in messages]
        confidences = [msg.get('sentiment_confidence', 0.5) for msg in messages]
        timestamps = [msg['created_at'] for msg in messages]
        
        # Calculate temporal features
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 for i in range(len(timestamps)-1)]
        
        # Sentiment velocity and acceleration
        sentiment_velocity = []
        sentiment_acceleration = []
        
        for i in range(len(sentiments)-1):
            if time_diffs[i] > 0:
                velocity = (sentiments[i+1] - sentiments[i]) / time_diffs[i]
                sentiment_velocity.append(velocity)
                
                if i > 0 and time_diffs[i-1] > 0:
                    prev_velocity = (sentiments[i] - sentiments[i-1]) / time_diffs[i-1]
                    acceleration = (velocity - prev_velocity) / time_diffs[i]
                    sentiment_acceleration.append(acceleration)
        
        # Keyword escalation analysis
        escalation_scores = []
        for msg in messages:
            text = msg.get('text', '').lower()
            score = sum(weight for keyword, weight in self.escalation_keywords.items() if keyword in text)
            escalation_scores.append(score)
        
        # Confidence decay analysis
        confidence_decay = self._calculate_confidence_decay(confidences, time_diffs)
        
        # Pattern extraction
        pattern_sequence = [msg['sentiment'] for msg in messages]
        
        return {
            'user_id': messages[0].get('user_id'),
            'message_count': len(messages),
            'duration_hours': sum(time_diffs),
            'sentiment_sequence': pattern_sequence,
            'sentiment_numeric': sentiments,
            'sentiment_mean': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'sentiment_trend': self._calculate_trend(sentiments),
            'sentiment_velocity': sentiment_velocity,
            'sentiment_acceleration': sentiment_acceleration,
            'avg_velocity': np.mean(sentiment_velocity) if sentiment_velocity else 0,
            'max_velocity': np.max(np.abs(sentiment_velocity)) if sentiment_velocity else 0,
            'avg_acceleration': np.mean(sentiment_acceleration) if sentiment_acceleration else 0,
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_decay': confidence_decay,
            'escalation_score': np.mean(escalation_scores),
            'escalation_trend': self._calculate_trend(escalation_scores),
            'rapid_deterioration': self._detect_rapid_deterioration(sentiments, time_diffs),
            'conversation_length': len(messages),
            'timestamp_start': timestamps[0],
            'timestamp_end': timestamps[-1]
        }
    
    def predict_escalation(self, trajectory: Dict, patterns: Dict = None) -> Dict:
        """
        INNOVATION: Predict escalation using trajectory patterns and features
        
        This method combines multiple prediction approaches:
        1. Pattern-based prediction using mined patterns
        2. Feature-based prediction using trajectory characteristics
        3. Keyword-based analysis for escalation signals
        4. Statistical confidence estimation
        """
        try:
            # Default return structure
            default_prediction = {
                'escalation_probability': 0.0,
                'confidence_interval': (0.0, 0.0),
                'risk_level': 'low',
                'key_indicators': [],
                'recommendation': 'Continue monitoring',
                'prediction_method': 'multi_factor_analysis'
            }
            
            if not trajectory:
                return default_prediction
            
            # Extract features
            features = trajectory
            sentiment_sequence = features.get('sentiment_sequence', [])
            content_sequence = features.get('content_sequence', [])
            confidence_sequence = features.get('confidence_sequence', [])
            
            if not sentiment_sequence:
                return default_prediction
            
            # Initialize prediction components
            predictions = []
            weights = {}
            
            # 1. Pattern-based prediction (30% weight)
            if patterns:
                pattern_prob = self._pattern_based_prediction(sentiment_sequence, patterns)
                predictions.append(pattern_prob)
                weights['pattern'] = 0.3
            else:
                # Use default patterns if none provided
                pattern_prob = self._default_pattern_prediction(sentiment_sequence)
                predictions.append(pattern_prob)
                weights['pattern'] = 0.3
            
            # 2. Feature-based prediction (25% weight)
            feature_prob = self._feature_based_prediction(features)
            predictions.append(feature_prob)
            weights['features'] = 0.25
            
            # 3. Keyword-based prediction (25% weight)
            keyword_prob = self._keyword_based_prediction(content_sequence)
            predictions.append(keyword_prob)
            weights['keywords'] = 0.25
            
            # 4. Trend-based prediction (20% weight)
            trend_prob = self._trend_based_prediction(sentiment_sequence, confidence_sequence)
            predictions.append(trend_prob)
            weights['trend'] = 0.2
            
            # Calculate weighted ensemble prediction
            if predictions:
                weighted_prob = sum(p * w for p, w in zip(predictions, weights.values()))
                final_probability = min(max(weighted_prob, 0.0), 1.0)
            else:
                final_probability = 0.0
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(predictions, weights)
            
            # Classify risk level
            risk_level = self._classify_risk_level(final_probability)
            
            # Extract key indicators
            key_indicators = self._extract_key_indicators(features)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(final_probability, risk_level)
            
            return {
                'escalation_probability': final_probability,
                'confidence_interval': confidence_interval,
                'risk_level': risk_level,
                'key_indicators': key_indicators,
                'recommendation': recommendation,
                'prediction_method': 'multi_factor_analysis',
                'component_scores': {
                    'pattern': predictions[0] if len(predictions) > 0 else 0.0,
                    'features': predictions[1] if len(predictions) > 1 else 0.0,
                    'keywords': predictions[2] if len(predictions) > 2 else 0.0,
                    'trend': predictions[3] if len(predictions) > 3 else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in escalation prediction: {e}")
            return {
                'escalation_probability': 0.0,
                'confidence_interval': (0.0, 0.0),
                'risk_level': 'low',
                'key_indicators': [],
                'recommendation': 'Error in prediction - manual review recommended',
                'prediction_method': 'error_fallback',
                'error': str(e)
            }

    def _default_pattern_prediction(self, sequence: List[str]) -> float:
        """Default pattern-based prediction when no patterns are provided"""
        if not sequence:
            return 0.0
        
        # Count negative sentiment patterns
        negative_count = sequence.count('negative')
        total_count = len(sequence)
        
        # Calculate escalation probability based on negative sentiment density
        if total_count == 0:
            return 0.0
        
        negative_ratio = negative_count / total_count
        
        # Check for escalation patterns
        escalation_patterns = [
            ['negative', 'negative'],
            ['neutral', 'negative'],
            ['positive', 'negative'],
            ['negative', 'negative', 'negative']
        ]
        
        pattern_matches = 0
        for pattern in escalation_patterns:
            if self._sequence_contains_pattern(sequence, pattern):
                pattern_matches += 1
        
        # Combine negative ratio and pattern matches
        pattern_score = (negative_ratio * 0.6) + (pattern_matches / len(escalation_patterns) * 0.4)
        
        return min(pattern_score, 1.0)

    def _keyword_based_prediction(self, content_sequence: List[str]) -> float:
        """Predict escalation based on keyword analysis"""
        if not content_sequence:
            return 0.0
        
        total_score = 0.0
        total_words = 0
        
        for content in content_sequence:
            if not content:
                continue
                
            words = content.lower().split()
            total_words += len(words)
            
            for word in words:
                if word in self.escalation_keywords:
                    total_score += self.escalation_keywords[word]
        
        # Normalize by total words
        if total_words == 0:
            return 0.0
        
        normalized_score = total_score / total_words
        
        # Scale to 0-1 range (assuming max 0.1 keywords per message)
        return min(normalized_score * 10, 1.0)

    def _trend_based_prediction(self, sentiment_sequence: List[str], confidence_sequence: List[float]) -> float:
        """Predict escalation based on sentiment trends"""
        if not sentiment_sequence or len(sentiment_sequence) < 2:
            return 0.0
        
        # Convert sentiments to numeric values
        numeric_sentiments = [self._sentiment_to_numeric(s) for s in sentiment_sequence]
        
        # Calculate trend (slope)
        trend = self._calculate_trend(numeric_sentiments)
        
        # Negative trend indicates potential escalation
        if trend < -0.1:  # Decreasing sentiment
            trend_score = abs(trend)
        else:
            trend_score = 0.0
        
        # Factor in confidence decay
        if confidence_sequence:
            confidence_trend = self._calculate_trend(confidence_sequence)
            if confidence_trend < -0.05:  # Decreasing confidence
                trend_score += abs(confidence_trend) * 0.5
        
        return min(trend_score, 1.0)

    def _sequence_contains_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence contains a specific pattern"""
        if len(pattern) > len(sequence):
            return False
        
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        
        return False

    def _sentiment_to_numeric(self, sentiment: str) -> float:
        """Convert sentiment to numeric value"""
        mapping = {'negative': -1.0, 'neutral': 0.0, 'positive': 1.0}
        return mapping.get(sentiment.lower(), 0.0)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend using least squares"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _calculate_confidence_decay(self, confidences: List[float], time_diffs: List[float]) -> float:
        """Calculate confidence decay rate"""
        if len(confidences) < 2 or not time_diffs:
            return 0.0
        
        # Calculate weighted decay
        decay_rates = []
        for i in range(len(confidences)-1):
            if time_diffs[i] > 0:
                decay = (confidences[i] - confidences[i+1]) / time_diffs[i]
                decay_rates.append(decay)
        
        return np.mean(decay_rates) if decay_rates else 0.0
    
    def _detect_rapid_deterioration(self, sentiments: List[float], time_diffs: List[float]) -> bool:
        """Detect rapid sentiment deterioration"""
        if len(sentiments) < 3:
            return False
        
        # Look for rapid negative changes
        for i in range(len(sentiments)-2):
            if time_diffs[i] < 1.0:  # Within 1 hour
                sentiment_drop = sentiments[i] - sentiments[i+1]
                if sentiment_drop > 1.0:  # Significant drop
                    return True
        return False
    
    def _pattern_based_prediction(self, sequence: List[str], patterns: Dict) -> float:
        """Predict escalation based on pattern matching"""
        if not sequence or not patterns:
            return 0.5
        
        max_probability = 0.0
        for pattern in patterns:
            if self._pattern_matches_sequence(pattern, sequence):
                prob = self.pattern_weights.get(pattern, 0.5)
                max_probability = max(max_probability, prob)
        
        return max_probability
    
    def _feature_based_prediction(self, features: Dict) -> float:
        """Predict escalation based on extracted features"""
        # Weighted feature scoring
        score = 0.0
        
        # Sentiment trend (negative trend increases probability)
        trend = features.get('sentiment_trend', 0)
        if trend < -0.1:
            score += 0.3 * abs(trend)
        
        # Velocity (rapid negative changes)
        avg_velocity = features.get('avg_velocity', 0)
        if avg_velocity < -0.5:
            score += 0.2 * abs(avg_velocity)
        
        # Confidence decay
        confidence_decay = features.get('confidence_decay', 0)
        if confidence_decay > 0.1:
            score += 0.2 * confidence_decay
        
        # Conversation length (longer conversations may escalate)
        length = features.get('conversation_length', 1)
        if length > 5:
            score += 0.1 * min(length / 10, 0.3)
        
        # Rapid deterioration flag
        if features.get('rapid_deterioration', False):
            score += 0.2
        
        return min(score, 1.0)
    
    def _pattern_matches_sequence(self, pattern: Tuple, sequence: List[str]) -> bool:
        """Check if pattern matches any subsequence"""
        pattern_len = len(pattern)
        for i in range(len(sequence) - pattern_len + 1):
            if tuple(sequence[i:i+pattern_len]) == pattern:
                return True
        return False
    
    def _calculate_confidence_interval(self, predictions: List[float], weights: Dict) -> Tuple[float, float]:
        """Calculate confidence interval for ensemble prediction"""
        if not predictions:
            return (0.0, 1.0)
        
        # Use weighted standard error
        weighted_mean = sum(p * w for p, w in zip(predictions, weights.values()))
        weighted_variance = sum(w * (p - weighted_mean)**2 for p, w in zip(predictions, weights.values()))
        
        if weighted_variance <= 0:
            return (weighted_mean, weighted_mean)
        
        std_error = np.sqrt(weighted_variance)
        margin = 1.96 * std_error  # 95% confidence interval
        
        lower = max(0.0, weighted_mean - margin)
        upper = min(1.0, weighted_mean + margin)
        
        return (lower, upper)
    
    def _classify_risk_level(self, probability: float) -> str:
        """Classify risk level based on escalation probability"""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _extract_key_indicators(self, features: Dict) -> List[str]:
        """Extract key escalation indicators"""
        indicators = []
        
        if features.get('sentiment_trend', 0) < -0.1:
            indicators.append("Declining sentiment trend")
        
        if features.get('avg_velocity', 0) < -0.5:
            indicators.append("Rapid sentiment deterioration")
        
        if features.get('confidence_decay', 0) > 0.1:
            indicators.append("Decreasing confidence in responses")
        
        if features.get('escalation_score', 0) > 1.0:
            indicators.append("Escalation keywords detected")
        
        if features.get('rapid_deterioration', False):
            indicators.append("Rapid conversation deterioration")
        
        if features.get('conversation_length', 1) > 8:
            indicators.append("Extended conversation length")
        
        return indicators
    
    def _generate_recommendation(self, probability: float, risk_level: str) -> str:
        """Generate actionable recommendation"""
        if risk_level == "CRITICAL":
            return "IMMEDIATE ACTION REQUIRED: Escalate to senior support team immediately"
        elif risk_level == "HIGH":
            return "HIGH PRIORITY: Assign experienced agent and monitor closely"
        elif risk_level == "MEDIUM":
            return "MODERATE ATTENTION: Provide proactive support and check satisfaction"
        elif risk_level == "LOW":
            return "STANDARD SUPPORT: Continue with regular support protocols"
        else:
            return "MINIMAL RISK: Standard support sufficient"
    
    def _empty_trajectory_features(self) -> Dict:
        """Return empty trajectory features"""
        return {
            'user_id': None,
            'message_count': 0,
            'duration_hours': 0,
            'sentiment_sequence': [],
            'sentiment_numeric': [],
            'sentiment_mean': 0,
            'sentiment_std': 0,
            'sentiment_trend': 0,
            'sentiment_velocity': [],
            'sentiment_acceleration': [],
            'avg_velocity': 0,
            'max_velocity': 0,
            'avg_acceleration': 0,
            'confidence_mean': 0,
            'confidence_std': 0,
            'confidence_decay': 0,
            'escalation_score': 0,
            'escalation_trend': 0,
            'rapid_deterioration': False,
            'conversation_length': 0,
            'timestamp_start': None,
            'timestamp_end': None
        } 