# filename: trade_logger.py

import csv
import os
from datetime import datetime

from analytics.logger import log


class TradeLogger:


    def __init__(
            self,
            folder="logs"
    ):

        self.folder = folder


        os.makedirs(
            folder,
            exist_ok=True
        )


        self.txt_file = os.path.join(
            folder,
            "trade_history.txt"
        )


        self.csv_file = os.path.join(
            folder,
            "trade_history.csv"
        )


        self._init_csv()


        log(
            "INFO | Trade Logger initialized"
        )



    # ======================================
    # CSV HEADER
    # ======================================

    def _init_csv(self):


        if not os.path.exists(
            self.csv_file
        ):


            with open(
                self.csv_file,
                "w",
                newline=""
            ) as f:


                writer = csv.writer(f)


                writer.writerow(
                    [
                        "symbol",
                        "type",
                        "entry_price",
                        "exit_price",
                        "volume",
                        "profit",
                        "open_time",
                        "close_time",
                        "reason",
                        "duration"
                    ]
                )



    # ======================================
    # SAVE CLOSED TRADE
    # ======================================

    def save_trade(
            self,
            trade
    ):


        try:


            open_time = trade.get(
                "open_time"
            )


            close_time = trade.get(
                "close_time"
            )


            if isinstance(open_time, int):

                open_time = datetime.fromtimestamp(
                    open_time
                )


            if isinstance(close_time, int):

                close_time = datetime.fromtimestamp(
                    close_time
                )


            duration = close_time - open_time


            duration_text = str(
                duration
            )



            # -------------------------
            # TXT REPORT
            # -------------------------

            with open(
                self.txt_file,
                "a"
            ) as f:


                f.write(
                    "\n"
                    "=================================\n"
                )


                f.write(
                    f"Symbol : {trade['symbol']}\n"
                )


                f.write(
                    f"Direction : {trade['type']}\n"
                )


                f.write(
                    f"Entry Price : {trade['entry_price']}\n"
                )


                f.write(
                    f"Exit Price : {trade.get(
                        "current_price",
                        trade.get("exit_price")
                    )}\n"
                )


                f.write(
                    f"Volume : {trade['volume']}\n"
                )


                f.write(
                    f"Profit : {trade['profit']} USD\n"
                )


                f.write(
                    f"Open Time : {open_time}\n"
                )


                f.write(
                    f"Close Time : {close_time}\n"
                )


                f.write(
                    f"Reason : {trade['exit_reason']}\n"
                )


                f.write(
                    f"Duration : {duration_text}\n"
                )


                f.write(
                    "=================================\n"
                )




            # -------------------------
            # CSV REPORT
            # -------------------------


            with open(
                self.csv_file,
                "a",
                newline=""
            ) as f:


                writer = csv.writer(f)


                writer.writerow(
                    [

                        trade["symbol"],

                        trade["type"],

                        trade["entry_price"],

                        trade["current_price"],

                        trade["volume"],

                        trade["profit"],

                        open_time,

                        close_time,

                        trade["exit_reason"],

                        duration_text

                    ]
                )



            log(
                f"INFO | Trade saved {trade['symbol']}"
            )



        except Exception as e:


            log(
                f"ERROR | Trade logging failed {e}"
            )