using UnityEngine;
using UnityEngine.AI;

public class PedestrianPatrol : MonoBehaviour
{
    public Transform[] waypoints;

    private NavMeshAgent agent;
    private int index;

    void Awake()
    {
        agent = GetComponent<NavMeshAgent>();
        agent.autoBraking = false;
        RestartPatrol();
    }

    void GoNext()
    {
        if (waypoints == null || waypoints.Length == 0) return;
        if (agent == null || !agent.isOnNavMesh) return;

        agent.destination = waypoints[index].position;
        index = (index + 1) % waypoints.Length;
    }

    public void RestartPatrol()
    {
        if (agent == null) agent = GetComponent<NavMeshAgent>();
        if (agent == null || !agent.isOnNavMesh) return;

        index = 0;
        agent.ResetPath();
        GoNext();
    }

    void Update()
    {
        if (agent == null || !agent.isOnNavMesh) return;

        if (!agent.pathPending && agent.remainingDistance < 0.2f)
            GoNext();
    }
}