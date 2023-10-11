from logger import Logger
from google.protobuf.duration_pb2 import Duration
from Declarations.Model.Report_pb2 import Report
from Declarations.Model.AIProcessingService import AIProcessingResponse_pb2


class ReportGenerator:
    def __init__(self, private_interface_client):
        self.log = Logger.get_logger(__name__)
        self.private_interface_client = private_interface_client

    def handle_finalize(self, request, client_token):
        """Handles the finalize request, generates a report, and sends it to the private interface client.

        Args:
            request: The finalize request.
            client_token: The client token used for authentication.

        Returns:
            AIProcessingResponse: The constructed response containing the generated report.
        """
        self.log.info(f"Handling Finalize request with reason: {request.finalize.finalize_reason}")
        report = self._create_report_feedback(3.9, "AI comment", 20, 20, 2.9)
        self.private_interface_client.report_processing_result(report, client_token)
        return AIProcessingResponse_pb2.AIProcessingResponse(report=report)

    def _create_report_feedback(self, avg_score, ai_comment, start_sec, end_sec, report_avg_score):
        """Creates a report feedback based on the provided parameters.

        Args:
            avg_score (float): The average score for the report.
            ai_comment (str): The AI-generated comment for the report.
            start_sec (int): The start time (in seconds) of the report segment.
            end_sec (int): The end time (in seconds) of the report segment.
            report_avg_score (float): The average score for the entire report.

        Returns:
            Report: The constructed report feedback.
        """
        report = Report()
        if not report.feedback:
            report.feedback.add()
        feedback_item = report.feedback[0]
        feedback_item.average_score = avg_score
        feedback_item.ai_comment = ai_comment
        feedback_item.start_time.CopyFrom(Duration(seconds=start_sec))
        feedback_item.end_time.CopyFrom(Duration(seconds=end_sec))
        report.average_score = report_avg_score
        return report
