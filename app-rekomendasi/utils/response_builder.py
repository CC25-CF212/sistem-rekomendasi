from flask import jsonify

class ResponseBuilder:
    @staticmethod
    def success(message, data=None, status_code=200):
        response = {
            "status": "success",
            "message": message,
            "data": data
        }
        return jsonify(response), status_code

    @staticmethod
    def error(message, details=None, status_code=400):
        response = {
            "status": "error",
            "message": message
        }
        if details:
            response["details"] = details
        return jsonify(response), status_code
