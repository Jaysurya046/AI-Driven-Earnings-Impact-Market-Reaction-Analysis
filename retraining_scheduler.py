"""
Automated Retraining Scheduler
Sets up scheduled retraining of ML models
"""

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Manages automated model retraining schedules"""
    
    def __init__(self, config_path: str = 'retraining_config.json'):
        """
        Initialize scheduler
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.last_training = None
        self.training_history = []
    
    def _load_config(self) -> dict:
        """Load or create default configuration"""
        default_config = {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
            "schedule": {
                "frequency": "weekly",  # daily, weekly, monthly
                "day_of_week": "monday",  # for weekly
                "day_of_month": 1,  # for monthly
                "time": "02:00"  # 2 AM UTC
            },
            "pipeline_config": {
                "lookback_years": 5,
                "event_window": 5,
                "estimation_window": 60,
                "label_threshold": 0.0
            },
            "retraining_triggers": {
                "min_data_points": 50,
                "max_model_age_days": 30,
                "performance_drop_threshold": 0.05
            },
            "notification": {
                "enabled": False,
                "email": None,
                "webhook_url": None
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        else:
            self._save_config(default_config)
            logger.info(f"Created default configuration at {self.config_path}")
        
        return default_config
    
    def _save_config(self, config: dict) -> None:
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {self.config_path}")
    
    def run_retraining(self, tickers: Optional[List[str]] = None) -> dict:
        """
        Execute model retraining
        
        Args:
            tickers: List of tickers to analyze (uses config if None)
            
        Returns:
            Dictionary with retraining results
        """
        if tickers is None:
            tickers = self.config.get('tickers', [])
        
        logger.info(f"Starting retraining for tickers: {tickers}")
        start_time = datetime.now()
        
        try:
            from main import EarningsAnalysisPipeline
            
            pipeline_config = self.config.get('pipeline_config', {})
            pipeline_config.update({
                'data_dir': 'data',
                'models_dir': 'models',
                'viz_dir': 'visualizations'
            })
            
            pipeline = EarningsAnalysisPipeline(pipeline_config)
            
            # Run complete pipeline
            pipeline.run_complete_pipeline(tickers)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'timestamp': str(start_time),
                'duration_seconds': duration,
                'tickers': tickers,
                'events_analyzed': len(pipeline.event_analysis) if pipeline.event_analysis is not None else 0
            }
            
            self.last_training = start_time
            self.training_history.append(result)
            
            logger.info(f"Retraining completed successfully in {duration:.2f} seconds")
            
            # Send notification if enabled
            if self.config.get('notification', {}).get('enabled'):
                self._send_notification('success', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}", exc_info=True)
            
            result = {
                'status': 'failed',
                'timestamp': str(start_time),
                'error': str(e)
            }
            
            # Send failure notification
            if self.config.get('notification', {}).get('enabled'):
                self._send_notification('failure', result)
            
            return result
    
    def _send_notification(self, status: str, result: dict) -> None:
        """Send notification about training completion"""
        notification_config = self.config.get('notification', {})
        
        # Email notification (requires additional setup)
        if notification_config.get('email'):
            logger.info(f"Would send email to {notification_config['email']}")
            # TODO: Implement email sending
        
        # Webhook notification
        if notification_config.get('webhook_url'):
            import requests
            try:
                requests.post(
                    notification_config['webhook_url'],
                    json={'status': status, 'result': result},
                    timeout=10
                )
                logger.info("Webhook notification sent")
            except Exception as e:
                logger.error(f"Failed to send webhook: {e}")
    
    def check_retraining_needed(self) -> bool:
        """
        Check if retraining is needed based on triggers
        
        Returns:
            True if retraining is needed
        """
        triggers = self.config.get('retraining_triggers', {})
        
        # Check model age
        max_age = triggers.get('max_model_age_days', 30)
        if self.last_training:
            days_since_training = (datetime.now() - self.last_training).days
            if days_since_training >= max_age:
                logger.info(f"Model age ({days_since_training} days) exceeds threshold ({max_age} days)")
                return True
        
        # Check data availability
        data_dir = Path('data')
        events_file = data_dir / 'event_analysis.csv'
        if events_file.exists():
            import pandas as pd
            df = pd.read_csv(events_file)
            min_points = triggers.get('min_data_points', 50)
            if len(df) >= min_points:
                # Check if we have significantly more data
                logger.info(f"Data points ({len(df)}) meets threshold ({min_points})")
                # Could add logic to check if enough new data has accumulated
        
        return False
    
    def setup_schedule(self):
        """Set up the retraining schedule based on configuration"""
        schedule_config = self.config.get('schedule', {})
        frequency = schedule_config.get('frequency', 'weekly')
        time_str = schedule_config.get('time', '02:00')
        
        logger.info(f"Setting up {frequency} retraining schedule at {time_str}")
        
        if frequency == 'daily':
            schedule.every().day.at(time_str).do(self.run_retraining)
            logger.info("Scheduled daily retraining")
        
        elif frequency == 'weekly':
            day = schedule_config.get('day_of_week', 'monday').lower()
            day_methods = {
                'monday': schedule.every().monday,
                'tuesday': schedule.every().tuesday,
                'wednesday': schedule.every().wednesday,
                'thursday': schedule.every().thursday,
                'friday': schedule.every().friday,
                'saturday': schedule.every().saturday,
                'sunday': schedule.every().sunday
            }
            getattr(day_methods.get(day, schedule.every().monday), 'at')(time_str)(self.run_retraining)
            logger.info(f"Scheduled weekly retraining on {day} at {time_str}")
        
        elif frequency == 'monthly':
            # Monthly requires custom handling
            day_of_month = schedule_config.get('day_of_month', 1)
            logger.info(f"Monthly scheduling requires custom implementation (day {day_of_month})")
            # For simplicity, we'll use weekly and check date in run_retraining
        
        else:
            logger.warning(f"Unknown frequency: {frequency}. No schedule set.")
    
    def run_continuous(self, check_interval_minutes: int = 60):
        """
        Run scheduler continuously
        
        Args:
            check_interval_minutes: How often to check for scheduled tasks
        """
        self.setup_schedule()
        
        logger.info(f"Starting continuous scheduler (checking every {check_interval_minutes} minutes)")
        
        while True:
            schedule.run_pending()
            
            # Also check retraining triggers periodically
            if self.check_retraining_needed():
                logger.info("Retraining triggers detected, running retraining")
                self.run_retraining()
            
            time.sleep(check_interval_minutes * 60)


def main():
    """Main entry point for scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Model Retraining Scheduler')
    parser.add_argument('--run-now', action='store_true', help='Run retraining immediately')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous mode')
    parser.add_argument('--config', default='retraining_config.json', help='Config file path')
    
    args = parser.parse_args()
    
    scheduler = RetrainingScheduler(config_path=args.config)
    
    if args.run_now:
        result = scheduler.run_retraining(tickers=args.tickers)
        print(json.dumps(result, indent=2))
    elif args.continuous:
        scheduler.run_continuous()
    else:
        print("Use --run-now to trigger immediate retraining")
        print("Use --continuous to run the scheduler daemon")
        print("Use --help for more options")


if __name__ == "__main__":
    main()
