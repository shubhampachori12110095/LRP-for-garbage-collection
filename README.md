# LRP-for-garbage-collection
Two-Echelon Capacitated location-routing problem of Heterogeneous fleets
## Model
### parameter set

| Vehicle parameters  |                         description                          |                         value                         |
| :-----------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|      $\kappa$       |              Engine friction factor  ($kj/rev/l$)              | 0.2 |
|          $N$          |                     Engine speed ($rev/s$)                     | 36.67 |
|          $V$          |                   Engine displacement  ($l$)                   | 6.9 |
|     $\epsilon$      |               Vehicle drive train  efficiency                | 0.45 |
|        $P_a$        | engine power demand associated  with running losses of the engine and additional vehicle accessories | 0 |
|       $\eta$        |                efficiency for diesel  engines                | 0.45 |
|      $\omega$       |                        Curb weigh($kg$)                        | 5500 |
|          $a$          |                    Acceleration ($m/s^2$)                    | 0 |
|          $v$          |                        Speed($m/s$ )                         | 15 |
|          $g$          |              Gravitational constant  ($m/s^2$)               | 9.8 |
|      $\theta$       |                          Road angle                          | 0 |
|          $A$          |                 Frontal surface area ($m^2$)                 | 8.0 |
|       $\rho$        |                    Air density ($kg/m^3$)                    | 1.2 |
|        $C_d$        |              Coefficient of  aerodynamics drag               | 0.7 |
|        $C_r$        |              Coefficient of rolling  resistance              | 0.01 |
|          $U$          |     a value that depends  on some constants including N.     | 0.025 |
|        $C_f$        |                      unit cost of fuel                       | 6.5 |
| $\delta_1,\delta_2$ |           GHG-specific emission  index parameters            |                       |



| Sets  |                         description                          |
| :---: | :----------------------------------------------------------: |
|   $K$   | The set of vehicle  types operated between transfer station and collection point $k\in K$ |
| $M_k$ |         The set of vehicles of  type k ,$m_k\in M_k$         |
| $M_g$ | The set of vehicle $m_g\in M_g$   depart from garbage  treatment plant $g\in G$ |
|   $G$   |         The set of garbage  treatment plant $g\in G$         |
|   $T$   |            The set of transfer  station $t\in T$             |
|   $C$   |            The set of collection  point $c\in C$             |
| $N_1$ |         The set of nodes  consist of $\{ G\cup T\}$          |
| $N_2$ |         The set of nodes  consist of $\{ T\cup C\}$          |
| $TR$  |               The set of trash type $r\in TR$                |



|    parameters    |                         description                          |
| :--------------: | :----------------------------------------------------------: |
|      $OC_t$      |        Operation cost of a  transfer station $t\in T$        |
|     $d_{ij}$     |               Distance from node i  to node j                |
|      $FV_k$      |               Fixed cost of vehicle  type of k               |
|      $FV_m$      |               Fixed cost of vehicle  type of m               |
|    $de_{cr}$     | Demand of trash of  type of $r\in TR$ for collection point collection point $c\in C$ |
|    $de_{tr}$     | Demand of trash of  type of $r\in TR$ for transfer station $t\in T$ |
|     $CAP_m$      |            Capacity of vehicle  type of $m\in M$             |
|    $CAP_{kr}$    | Capacity of trash of  type $r\in TR$ for vehicle type $k\in K$ |
|  $CA{{P}_{t}}$   |            Capacity of transfer  station $t\in T$            |
|        $MT$        |                     Maximum travel time                      |
|      $ST_c$      |          Service time for  collection point$c\in C$          |
| $f_{ijr}^{m_kk}$ | The weight of trash  type $r\in TR$between node $\text{i,  }j\in {{N}_{2}}$ by vehicle $m_k$ of type k |
|  $e_{ijr}^{m_g}$  | The weight of trash  type $r\in TR$between node $\text{i,  }j\in {{N}_{1}}$ by vehicle $m\in M$ |
|    $dis_{gr}$    | $\begin{cases}  \mbox{1 if garbage  treatment plant } g\in G \mbox{dispose the type of  garbage } t\in TR\\0\  \mbox{ otherwise}  \end{cases}$ |



|    variable     |                         description                          |
| :-------------: | :----------------------------------------------------------: |
|      $y_t$      | $\begin{cases}  \mbox{1 if transfer  station } t\in T \mbox{ is opened}\\0 \mbox{ otherwise}     \end{cases}$ |
|    $h_{tc}$     | $\begin{cases} \mbox{1 if collection point } c\in C \mbox{ is served by  transfer station }t\in T\\0 \mbox{ otherwise}  \end{cases}$ |
| $x_{ij}^{m_kk}$ | $\left\{ \begin{array}{*{35}{l}}    1 \mbox{ if vehicle } m_k \mbox{traverse arc }\{i,j\}\in {{N}_{2}} \\    0\mbox{ otherwise} \\  \end{array} \right.$ |
| $u_{ij}^{m_g}$  | $\left\{ \begin{array}{*{35}{l}}    \mbox{1 if vehicle m depart from } g\in G \mbox{ traverse arc}\{i,j\ \}\in  {{N}_{1}} \\    0\ \mbox{otherwise} \\  \end{array} \right.$ |



### objective function analysis

Fuel consumption rate F:

$F\approx (\kappa NV+(P_{t}/\epsilon +P_{a})/\eta)U$

​	$\kappa$ is Engine friction factor, $N$ is Engine speed, $V$ is Engine displacement, $\epsilon$ Vehicle drive train  efficiency, $P_a$ is engine power demand associated  with running losses of the engine and additional vehicle accessories, $\eta$ efficiency for diesel  engines, $U$ a value that depends  on some constants including N. $P_t$ is the total tractive power demand requirement in watts.

We assume $P_{a}$ = 0

$F\approx (\kappa NV+({{P}_{t}}/\epsilon )/\eta )U$

${{P}_{t}}=(Ma+Mgsin\theta +0.5{{C}_{d}}A\rho {{v}^{2}}+Mg{{C}_{r}}cos\theta )v$/1000 (kW)

​	$M$ is the mass (kg) of the vehicle (empty plus carried load), v is speed, a is the acceleration, g is the gravitational
constant $(9.81 m/s2)$, h is the road angle, A is the frontal surface area of the vehicle, q is the air density,
and Cr and Cd are the coefficients of rolling resistance and drag, respectively

Assume $\alpha=0.5C_{d}A\rho v^{2}=0.5*0.7*8.0*1.2*15^2=756$,$\beta = a+gsin\theta+gC_{r}cos\theta=0+0+9.8*0.01=0.098$,$P_{t} = M\beta v+\alpha v=1.47M+11340$, where $M=\omega +f_{ij}=5500+f_{ij}$

Let $\gamma =1/1000\epsilon \eta=\frac{1}{1000*0.45*0.45}=0.005 $

$F=U(\kappa NV+M\beta v\gamma+\alpha v\gamma)$

The instantaneous engine-out emission rate E in grams per second ($kg/s$) for a GHG (such as CO, HC or NOx):

$E=\delta_{1}F+\delta_{2}$

**Total fuel consumption over a distance d is calculated as:**

$F_{total}=U(\kappa NVd/v+M\beta \gamma d+\alpha \gamma d)$ (kg)=$0.025(0.2*36.67*6.9d/15+(5500+f_{ij})*0.098*0.005d+756*0.005d)$$=0.025*(3.37364+0.00049(5500+f_{ij})+3.78)d$

**Total transportation cost:**

**$TC=c_{f}F_{total}$**

**Total emission:**

**$EC=\delta_{1}F_{total}+\delta_{2}$**

### model:

$$
Min1=
\sum_{i,j\in N_1}\sum_{g\in G}\sum_{m_g\in M_g}c_fU_1(\kappa_1N_1V_1d_{ij}/v_1+\alpha_1\gamma_1d_{ij}+\omega_1\beta_1\gamma_1d_{ij})u_{ij}^{m_g}+\sum_{i,j\in N_1}\sum_{g\in G}\sum_{m_g\in M_g}\sum_{r\in TR}e_{ijr}^{m_g}\beta_1\gamma_1d_{ij}\\
+\sum_{i,j\in N_2}\sum_{k\in K}\sum_{m_k\in M_k}c_fU_2(\kappa _2N_2V_2d_{ij}/v_2+\alpha_2\gamma_2d_{ij}+\omega_2\beta_2\gamma _2d_{ij})x_{ij}^{m_kk}+\sum_{i,j\in N_2}\sum_{k\in K}\sum_{m_k\in M_k}\sum_{r\in TR}f_{ijr}^{m_kk}\beta_2\gamma _2d_{ij}\\

+\sum_{i\in T}\sum_{j\in C}\sum_{k\in K}\sum_{m_k\in M_k}FV_kx_{ij}^{m_kk}+\sum_{i\in G}\sum_{j\in T}\sum_{m_i\in M_i}FV_mu_{ij}^{m_i}+\sum_{t\in T}OC_ty_t
$$

<a name='obj1'> </a>


unchanged objective Ⅱ
$$
Min2=
\sum\limits_{i,j\in N_1}\sum\limits_{m\in M} (\delta_1U_1(\kappa_1N_1V_1d_{ij}/v_1+(\omega_1+\sum\limits_{r\in TR}e_{ijr}^m)\beta_1\gamma_1d_{ij})+\delta_2)u_{ij}^m\\
+\sum\limits_{i,j\in {{N}_{1}}}{\sum\limits_{k\in K}{\sum\limits_{{{m}_{k}}\in {{M}_{k}}}{({{\delta }_{1}}{{U}_{2}}(}}}{{\kappa }_{2}}{{N}_{2}}{{V}_{2}}{{d}_{ij}}/{{v}_{2}}+({{\omega }_{2}}+\sum\limits_{r\in TR}{f_{ijr}^{{{m}_{k}}k}}){{\beta }_{2}}{{\gamma }_{2}}{{d}_{ij}})+{{\delta }_{2}})x_{ij}^{{{m}_{k}}k}
$$
 <a name='obj2'> </a>



​                                   

------

Subject to:
<a name='con3'> </a>
$$
&\sum\limits_{i\in {{N}_{2}}}{\sum\limits_{{{m}_{k}}\in {{M}_{k}}}{x_{ij}^{{{m}_{k}}k}}}=1\ \ \ \forall j\in C,\ k\in K\\
$$
<a name='con4'> </a>each trash type in each collection point is visited once
$$
&\sum\limits_{i\in {{N}_{2}}}{x_{ij}^{{{m}_{k}}k}}=\sum\limits_{i\in {{N}_{2}}}{x_{ji}^{{{m}_{k}}k}}\ \ \ \forall j\in {{N}_{2}},\ {{m}_{k}}\in {{M}_{k}},\ k\in K\\
$$
<a name='con5'> </a>degree constraint
$$
&x_{ij}^{{{m}_{k}}k}\le {{h}_{ij}}\ \ \ \forall i\in T,\ j\in C,\ {{m}_{k}}\in {{M}_{k}},k\in K\\
$$
<a name='con6'> </a>constraint(5)-(7) denote that vehicle depart from and end at the same transfer station and all the visited collection points are assigned to this transfer station. of which means that if there is a route between i and j, i and j are assigned to the same transfer station
$$
&x_{ji}^{m_kk}\le h_{ij}\ \ \ \forall i\in T,\ j\in C,\ m_k\in M_k,\ k\in K\\
$$
<a name='con7'> </a>
$$
&x_{ij}^{m_kk}+h_{ip}+\sum_{q\in T,\ q\ne p}h_{jq}\le 2\ \ \ \forall (i,j)\in C,\ p\in T\\
$$
<a name='con8'> </a>
$$
&\sum_{i\in T}\sum_{j\in C}x_{ij}^{m_kk}\le 1\ \ \ \forall m_k\in M_k,\ k\in K\\
$$
<a name='con9'> </a>each vehicle can leave at most one transfer station
$$
\sum_{i\in N_2}\sum_{j\in C}de_{jr}x_{ij}^{m_kk}\le CAP_{kr}\ \ \ \forall r\in TR,\ m_k\in M_k,\ k\in K\\ no\ need
$$
<a name='con10'> </a>the total amount of waste at the collection point reached by the collection vehicle is less than the capacity of the vehicle
$$
&\sum\limits_{i\in C}{\sum\limits_{r\in TR}{{{h}_{ti}}}}d{{e}_{ir}}\le CA{{P}_{t}}{{y}_{t}}\ \ \forall t\in T\\
$$
<a name='con11'> </a>if transfer station t is not opening, no collection point will be assigned to it
$$
&\sum_{i\in N_2}\sum_{m_k\in M_k}\sum_{k\in K}f_{jir}^{m_kk} - \sum_{i\in N_2}\sum_{m_k\in M_k}\sum_{k\in K} f_{ijr}^{m_kk}=de_{jr}\ \ \ \forall j\in C,\ r\in TR\\
$$
<a name='con12'> </a>all the collection point has to be fully satisfied
$$
&\sum_{j\in C}\sum_{m_k\in M_k}\sum_{k\in K}\sum_{r\in TR}f_{ijr}^{m_kk}=0\ \ \ \forall i\in T\\
$$
<a name='con13'> </a>the outflow of the transfer station is 0
$$
&f_{ijr}^{m_kk}\le CAP_{kr}x_{ij}^{m_kk}\ \ \ \forall (i,j)\in N_2,\ r\in TR_k, m_k\in M_k,\ k\in K\\
$$
<a name='con14'> </a>if vehicle $m_k$ don't travel on route *ij*, the flow **f** is equal to 0
$$
&\sum_{t\in T}h_{tc}=1\ \ \ \forall c\in C\\
$$
<a name='con15'> </a>each collection point can only be assigned to one transfer station
$$
\sum_{(i,j)\in N_2}x_{ij}^{m_kk}d_{ij}/v+\sum_{i\in N_2}\sum_{j\in C}ST_jx_{ij}^{m_kk}\le MT\ \ \ \forall m_k\in M_k,\ k\in K
$$
<a name='con16'> </a>the total travelling time and service should be no more than the request in advance

-------------------------------------------
$$
\sum\limits_{i\in N_1}\sum\limits_{m_g\in M_g}u_{ij}^{m_g}={{y}_{\text{j}}}\quad\forall j\in T,\ g\in G
$$
<a name='con17'> </a>if transfer station j is opening, it will be visited once by semi-truck from each garbage treatment plant
$$
\sum_{i\in N_1}u_{ij}^{m_g}=\sum_{i\in N_1}u_{ji}^{m_g}\ \ \ \forall j\in T,g\in G,m_g\in M_g
$$
<a name='con18'> </a>degree constraint
$$
\sum_{j\in T}u_{gj}^{m_g}\le 1\ \ \ \forall g\in G,m_g\in M_g
$$
<a name='con19'> </a>each vehicle can leave at most one garbage treatment plant
$$
de_{jr}=\sum_{i\in C}h_{ji}de_{ir}\ \ \ \forall r\in TR,\ j\in T
$$
<a name='con20'> </a>the flow balance at transfer station
$$
\sum_{i\in N_1}\sum_{g\in G}\sum_{m_g\in M_g}dis_{gr}e_{jir}^{m_g}-\sum_{i\in N_1}\sum_{g\in G}\sum_{m_g\in M_g} dis_{gr}e_{ijr}^{m_g}=de_{jr}\ \ \ \forall j\in T,\ r\in TR
$$
<a name='con21'> </a>constraint(20)-(22) are the same type of constraints as [constraint(11)-(13)](#con11)
$$
\sum_{j\in T}\sum_{m_g\in M_g}\sum_{r\in TR}e_{ijr}^{m_g}=0\ \ \ \forall i\in G
$$
<a name='con22'> </a>
$$
e_{ijr}^{m_g}\le CAP_mu_{ij}^{m_g}\quad\forall (i,j)\in N_1,\ g\in G,\ m_g\in M_g,\ r\in TR
$$



$$
\sum_{i\in G,i\ne g}\sum_{j\in T}u_{ij}^{m_g}= 0\quad\forall g\in G,m_g\in M_g
$$


​	

#### constraint interpretation

​	[constraint(3)](#con3) denote that each trash type in each collection point is visited once,[constraint(4)](#con4) is a degree constraint. [constraint(5)-(7)](#con5) denote that vehicle depart from and end at the same transfer station and all the visited collection points are assigned to this transfer station. of which [constraint(7)](#con7) means that if there is a route between i and j, i and j are assigned to the same transfer station. [constraint(8)](#con8) requests that each vehicle can leave at most one transfer station. [constraint(9) and (10)](#con9) are capacity constraints, [constraint(9)](#con9) denote the total amount of waste at the collection point reached by the collection vehicle is less than the capacity of the vehicle,[constraint(10)](#con10) denote that if transfer station t is not opening, no collection point will be assigned to it. [constraint(11)-(13)](#con11) is flow constraints, for which [constraint(11)](#con11) forces all the collection point has to be fully satisfied, and [constraint(12)](#con12) denote the outflow of the transfer station is 0, [constraint(13)](#con13) links the **x** and **f**, which means that if vehicle $m_k$ don't travel on route *ij*, the flow **f** is equal to 0. [constraint(14)](#con14) denote that each collection point can only be assigned to one transfer station. [constraint(15)](#con15) is a time constraint, means that the total travelling time and service should be no more than the request in advance.

​	[constraint(16)](#con16 ) denote that if transfer station j is opening, it will be visited once by semi-truck from each garbage treatment plant. [constraint(17)](#con17) is a degree constraint. [constraint(18)](#con18) request that each vehicle can leave at most one garbage treatment plant. [constraint(19)](#con19) determined the flow balance at transfer station. [constraint(20)-(22)](#con20) are the same type of constraints as [constraint(11)-(13)](#con11)
