import axios from "axios";

import { graphDBConfig } from "../../../common/configs/graphdb.config";
import { LODMapper } from "./LODMapper";

export class GraphDBService {
  private readonly baseUrl: string;
  private readonly repoName: string;

  constructor() {
    this.baseUrl = graphDBConfig.baseUrl;
    this.repoName = graphDBConfig.repoName;
  }

  async getUserGraphData(userId: string): Promise<any> {
    const query = LODMapper.getSPARQLUserQuery(userId);
    try {
      const response = await axios.post(
        `${this.baseUrl}/repositories/${this.repoName}`,
        query,
        {
          headers: {
            "Content-Type": "application/sparql-query",
            Accept: "text/turtle", // Retrieve as Turtle format
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error(`❌ Failed to fetch graph data for user ${userId}:`, error);
      throw error;
    }
  }

  // Hàm gửi dữ liệu Turtle trực tiếp đến GraphDB
  async insertData(turtleData: string): Promise<void> {
    try {
      await axios.post(
        `${this.baseUrl}/repositories/${this.repoName}/statements`,
        turtleData,
        {
          headers: {
            "Content-Type": "text/turtle",
          },
        }
      );
      console.log("✅ Data pushed to GraphDB successfully");
    } catch (error) {
      console.error("❌ Failed to push to GraphDB:", error);
      throw error;
    }
  }
  async deleteUserData(userId: string): Promise<void> {
    const userUri = `http://omnimer.health/data/user_${userId}`;

    const sparqlUpdate = `
      PREFIX : <http://omnimer.health/data/>
      PREFIX sosa: <http://www.w3.org/ns/sosa/>
      PREFIX schema: <http://schema.org/>
      PREFIX fhir: <http://hl7.org/fhir/>

      DELETE {
        ?s ?p ?o .
      }
      WHERE {
        {
          BIND(<${userUri}> AS ?s)
          ?s ?p ?o .
        }
        UNION
        {
          ?s sosa:hasFeatureOfInterest <${userUri}> .
          ?s ?p ?o .
        }
        UNION
        {
          ?s schema:agent <${userUri}> .
          ?s ?p ?o .
        }
      }
    `;

    try {
      await axios.post(
        `${this.baseUrl}/repositories/${this.repoName}/statements`,
        sparqlUpdate,
        {
          headers: {
            "Content-Type": "application/sparql-update",
          },
        }
      );
      console.log(
        `✅ Data for user ${userId} deleted from GraphDB successfully`
      );
    } catch (error) {
      console.error(
        `❌ Failed to delete data for user ${userId} from GraphDB:`,
        error
      );
    }
  }
  async updateUserData(userId: string, turtleData: string): Promise<void> {
    console.log(`🔄 Updating data for user ${userId}...`);
    try {
      // Xóa dữ liệu cũ trước
      await this.deleteUserData(userId);
      // Thêm dữ liệu mới
      await this.insertData(turtleData);
      console.log(`✅ Data for user ${userId} updated successfully`);
    } catch (error) {
      console.error(
        `❌ Failed to update data for user ${userId} in GraphDB:`,
        error
      );
    }
  }

  async deleteGoalData(goalId: string): Promise<void> {
    const goalUri = `http://omnimer.health/data/gl_${goalId}`;

    const sparqlUpdate = `
      DELETE {
        ?s ?p ?o .
      }
      WHERE {
        BIND(<${goalUri}> AS ?s)
        ?s ?p ?o .
      }
    `;

    try {
      await axios.post(
        `${this.baseUrl}/repositories/${this.repoName}/statements`,
        sparqlUpdate,
        {
          headers: {
            "Content-Type": "application/sparql-update",
          },
        }
      );
      console.log(
        `✅ Data for goal ${goalId} deleted from GraphDB successfully`
      );
    } catch (error) {
      console.error(
        `❌ Failed to delete data for goal ${goalId} from GraphDB:`,
        error
      );
    }
  }

  async deleteHealthProfileData(healthProfileId: string): Promise<void> {
    const hpUri = `http://omnimer.health/data/hp_${healthProfileId}`;

    const sparqlUpdate = `
      DELETE {
        ?s ?p ?o .
      }
      WHERE {
        BIND(<${hpUri}> AS ?s)
        ?s ?p ?o .
      }
    `;

    try {
      await axios.post(
        `${this.baseUrl}/repositories/${this.repoName}/statements`,
        sparqlUpdate,
        {
          headers: {
            "Content-Type": "application/sparql-update",
          },
        }
      );
      console.log(
        `✅ Data for health profile ${healthProfileId} deleted from GraphDB successfully`
      );
    } catch (error) {
      console.error(
        `❌ Failed to delete data for health profile ${healthProfileId} from GraphDB:`,
        error
      );
    }
  }

  async deleteWorkoutData(workoutId: string): Promise<void> {
    const wkUri = `http://omnimer.health/data/wk_${workoutId}`;

    const sparqlUpdate = `
      DELETE {
        ?s ?p ?o .
      }
      WHERE {
        BIND(<${wkUri}> AS ?s)
        ?s ?p ?o .
      }
    `;

    try {
      await axios.post(
        `${this.baseUrl}/repositories/${this.repoName}/statements`,
        sparqlUpdate,
        {
          headers: {
            "Content-Type": "application/sparql-update",
          },
        }
      );
      console.log(
        `✅ Data for workout ${workoutId} deleted from GraphDB successfully`
      );
    } catch (error) {
      console.error(
        `❌ Failed to delete data for workout ${workoutId} from GraphDB:`,
        error
      );
    }
  }
}
