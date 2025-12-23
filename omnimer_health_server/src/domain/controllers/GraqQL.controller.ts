import { Request, Response } from "express";
import { GraphDBService } from "../services/LOD/GraphDB.service";

export class GraqQLController {
  private graphDBService: GraphDBService;

  constructor() {
    this.graphDBService = new GraphDBService();
  }

  getUserGraphData = async (req: Request, res: Response) => {
    try {
      const { userId } = req.params;
      const data = await this.graphDBService.getUserGraphData(userId);

      // Return as Turtle file
      res.setHeader("Content-Type", "text/turtle");
      res.setHeader(
        "Content-Disposition",
        `attachment; filename=user_${userId}_data.ttl`
      );
      res.status(200).send(data);
    } catch (error) {
      console.error("Error fetching user graph data:", error);
      res.status(500).json({ message: "Error fetching graph data", error });
    }
  };
}
