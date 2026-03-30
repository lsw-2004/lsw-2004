using UnityEngine;

public class RobotController : MonoBehaviour
{
    [Header("最大速度设置")]
    public float maxLinearSpeed = 1.5f;     // m/s
    public float maxAngularSpeed = 300f;    // deg/s

    private Rigidbody rb;
    private float cmdV;   // m/s
    private float cmdW;   // deg/s

    public float CurrentLinearVelocity
    {
        get
        {
            if (rb == null) return 0f;
            Vector3 v = rb.velocity;
            v.y = 0f;
            return v.magnitude;
        }
    }

    /// <summary>
    /// 返回角速度（度/s），方便和 maxAngularSpeed 统一量纲
    /// </summary>
    public float CurrentAngularVelocityDeg
    {
        get
        {
            if (rb == null) return 0f;
            return rb.angularVelocity.y * Mathf.Rad2Deg;
        }
    }

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    /// <summary>
    /// 输入为归一化动作 [-1,1]
    /// vNorm -> 线速度
    /// wNorm -> 角速度
    /// </summary>
    public void SetAction(float vNorm, float wNorm)
    {
        vNorm = Mathf.Clamp(vNorm, -1f, 1f);
        wNorm = Mathf.Clamp(wNorm, -1f, 1f);

        cmdV = vNorm * maxLinearSpeed;
        cmdW = wNorm * maxAngularSpeed;
    }

    /// <summary>
    /// 直接用物理量设置动作（可选）
    /// </summary>
    public void SetActionPhysical(float linearSpeed, float angularSpeedDeg)
    {
        cmdV = Mathf.Clamp(linearSpeed, -maxLinearSpeed, maxLinearSpeed);
        cmdW = Mathf.Clamp(angularSpeedDeg, -maxAngularSpeed, maxAngularSpeed);
    }

    void FixedUpdate()
    {
        if (rb == null) return;

        Vector3 targetVel = transform.forward * cmdV;
        rb.velocity = new Vector3(targetVel.x, rb.velocity.y, targetVel.z);

        float wRad = cmdW * Mathf.Deg2Rad;
        rb.angularVelocity = new Vector3(0f, wRad, 0f);
    }

    public void Stop()
    {
        cmdV = 0f;
        cmdW = 0f;

        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }
}