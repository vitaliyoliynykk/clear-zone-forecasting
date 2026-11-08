import logging
from bson import ObjectId
from datetime import date


# 695ba5fd019aac8d6850dc43, 69c5954c1693c505a1c58ab5 - Bedroom (7 Jan - 14 May)

# 697a459a7eec4c420b0ef553, 6ac7891fc24c67af58e107b1 - Living Room (28 Jan - 1 Jul)
# {
#   "meta.module_id": {$in: [ObjectId(
#     "697a459a7eec4c420b0ef553"
#   ), ObjectId(
#     "6ac7891fc24c67af58e107b1"
#   )]},
#   "sensor": { $in: ["co2", "temp", "hum"]},
#   "ts": { $lt: new Date("2026-07-02T00:00:00Z") }
# }


from clear_zone_forecasting.database import check_mongodb_connection, get_database



logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Starting Clear Zone Forecasting application...")
    init_db()




def init_db() -> None:
    database = get_database()
    check_mongodb_connection()
    logger.info("Successfully connected to MongoDB database '%s'.", database.name)

    measurement_count = database["measurements"].count_documents({"meta.module_id": ObjectId("695ba5fd019aac8d6850dc43"), "sensor": 'co2'})
    logger.info(
        "Collection 'measurements' contains %s documents.", measurement_count
    )
